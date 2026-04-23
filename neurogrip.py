import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import serial
import numpy as np
import os
import csv
import json
import time
import logging
import threading
from datetime import datetime
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy
import warnings
warnings.filterwarnings("ignore")
 
# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("neurogrip.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("NeuroGrip")
 
print("\n" + "="*62)
print("    NeuroGrip v2.0 — Driver Stress & Fatigue Detection")
print("    Janavi Pawar 23BCB0041 | Hardik Teotia 23BCB0052")
print("="*62 + "\n")
 
# ==================== CONFIG ====================
CONFIG = {
    "port":           "COM5",
    "baud":           115200,
    "sample_rate":    20,        # Hz — must match ESP32
    "seq_len":        10,        # LSTM sequence length
    "fft_window":     20,        # FIX: was 60. 20 samples = 1sec @ 20Hz, GCI works from sample 20
    "lr":             0.005,
    "num_classes":    4,         # Relax, Stressed, Fatigued, Anomaly
    "retrain_every":  15,        # new samples before retrain trigger
    "alert_cooldown": 5,         # seconds between printed alerts
    "model_path":     "model_v2.pth",
    "data_path":      "data_v2.csv",
    "baseline_path":  "baseline.json",
    "session_path":   "session.json",
}
 
# Class labels — matching your detection logic
LABELS     = ["Relax", "Stressed", "Fatigued", "Anomaly"]
ALERT_LVLS = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
ALERT_ICONS = ["🟢", "🟡", "🟠", "🔴"]
 
 
# ==================== NEURAL NETWORK ====================
class NeuroGripNet(nn.Module):
    """
    Hybrid dual-branch network for grip biosignal classification.
 
    Branch 1 — LSTM:    Temporal grip dynamics (tremor drift, fatigue patterns)
    Branch 2 — FFN:     Instantaneous multi-sensor fusion (FSR + GSR + tremor)
    Branch 3 — Tremor:  FFT-derived features (GCI, spectral entropy, tremor band)
 
    Novel: Learnable fusion weights across branches.
    Novel: Confidence calibration auxiliary head for uncertainty quantification.
    """
    def __init__(self, input_size=3, lstm_h=24, ffn_h=32, num_classes=4, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
 
        # Branch 1: LSTM — captures temporal fatigue drift
        self.lstm = nn.LSTM(input_size, lstm_h, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.lstm_norm = nn.LayerNorm(lstm_h)
 
        # Branch 2: FFN — instantaneous sensor state
        self.ffn = nn.Sequential(
            nn.Linear(input_size, ffn_h),
            nn.GELU(),
            nn.LayerNorm(ffn_h),
            nn.Dropout(0.15),
            nn.Linear(ffn_h, ffn_h // 2),
            nn.GELU(),
        )
 
        # Branch 3: Tremor / frequency domain features
        # Input: [dom_freq, spec_entropy, tremor_ratio, GCI]
        self.tremor_branch = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8)
        )
 
        # Fusion (learnable weighted concat)
        fusion_in = lstm_h + ffn_h // 2 + 8
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 24),
            nn.GELU(),
            nn.LayerNorm(24),
        )
        self.classifier  = nn.Linear(24, num_classes)
        # Auxiliary confidence head (novel: uncertainty quantification)
        self.conf_head   = nn.Sequential(
            nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
        # Learnable branch fusion weights
        self.fw = nn.Parameter(torch.ones(3) / 3)
 
    def forward(self, x_seq, x_inst, x_trem):
        lstm_out, _ = self.lstm(x_seq)
        lf = self.lstm_norm(lstm_out[:, -1, :])
        ff = self.ffn(x_inst)
        tf = self.tremor_branch(x_trem)
 
        w  = F.softmax(self.fw, dim=0)
        fused = torch.cat([w[0]*lf, w[1]*ff, w[2]*tf], dim=-1)
        fused = self.fusion(fused)
 
        return self.classifier(fused), self.conf_head(fused)
 
 
# ==================== ONLINE GAUSSIAN BASELINE ====================
class PersonalizedBaseline:
    """
    Continuously updated Gaussian profile of each driver's grip signature.
 
    Novel: No fixed calibration window — updates every sample using EMA statistics.
    Anomaly score = Mahalanobis distance from personal distribution.
 
    Primary channel: GSR proxy (bio-impedance rises with stress/sweat)
    Secondary: FSR (grip pressure changes with fatigue)
    Tertiary: Tremor proxy
    """
    def __init__(self, n_channels=3, alpha=0.04):
        self.alpha = alpha
        self.mean  = np.zeros(n_channels)
        self.var   = np.ones(n_channels) * 0.01
        self.n     = 0
        self.ready = False
 
    def update(self, x):
        if not self.ready:
            self.mean = x.copy(); self.var = np.ones_like(x) * 0.01
            self.ready = True
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
            self.var  = (1 - self.alpha) * self.var  + self.alpha * (x - self.mean)**2
        self.n += 1
 
    def normalize(self, x):
        std = np.sqrt(self.var + 1e-6)
        return np.clip((x - self.mean) / std / 3.0, -1, 1) * 0.5 + 0.5
 
    def mahalanobis(self, x):
        std = np.sqrt(self.var + 1e-6)
        return float(np.sqrt(np.sum(((x - self.mean) / std) ** 2)))
 
    def save(self, path):
        json.dump({"mean": self.mean.tolist(), "var": self.var.tolist(), "n": self.n},
                  open(path, "w"))
 
    def load(self, path):
        d = json.load(open(path))
        self.mean = np.array(d["mean"])
        self.var  = np.array(d["var"])
        self.n    = d["n"]
        self.ready = True
 
 
# ==================== TREMOR ANALYZER (FFT + GCI) ====================
class TremorAnalyzer:
    """
    Real-time frequency-domain analysis of grip signal.
 
    Grip Coherence Index (GCI) — Novel Metric:
        GCI = (low_freq_power / total_power) / (spectral_entropy / log(N) + ε)
        GCI → 1.0 : Stable, intentional, alert grip
        GCI → 0.0 : Tremor-contaminated, fatigued or stressed grip
 
    Tremor band 3–8 Hz: Known biomarker for neuromotor fatigue precursors.
    Primary signal: Tremor channel (Ch 2), secondary: FSR (Ch 0).
    """
    def __init__(self, window=60, sr=20):
        self.window = window
        self.sr     = sr
        self.bufs   = [deque(maxlen=window) for _ in range(3)]
        self.freqs  = np.fft.rfftfreq(window, d=1.0/sr)
 
    def update(self, x):
        for i in range(3): self.bufs[i].append(x[i])
 
    def features(self):
        if len(self.bufs[0]) < self.window:  # FIX: use FSR (ch0), MPU6050 not connected
            return np.zeros(4)
 
        # FIX: Use FSR (Ch 0) for tremor — MPU6050 not connected yet
        sig = np.array(self.bufs[0]) - np.mean(self.bufs[0])
        fft_pwr = np.abs(np.fft.rfft(sig)) ** 2
        total   = fft_pwr.sum() + 1e-10
 
        # Tremor band 3–8 Hz
        tremor_mask = (self.freqs >= 3) & (self.freqs <= 8)
        tremor_pwr  = fft_pwr[tremor_mask].sum()
 
        # Low-frequency band ≤ 2 Hz (intentional grip)
        low_pwr = fft_pwr[self.freqs <= 2].sum()
 
        # Dominant frequency
        dom_freq = float(self.freqs[np.argmax(fft_pwr)])
 
        # Spectral entropy
        psd_norm   = fft_pwr / total
        spec_entr  = float(sp_entropy(psd_norm + 1e-10))
 
        # Tremor ratio
        tremor_ratio = float(tremor_pwr / total)
 
        # GCI — novel metric
        gci = (low_pwr / total) / (spec_entr / (np.log(max(len(psd_norm), 2)) + 1e-3) + 1e-3)
        gci = float(np.clip(gci, 0, 1))
 
        return np.array([dom_freq / 10.0, spec_entr / 5.0, tremor_ratio, gci])
 
 
# ==================== ADAPTIVE NOISE FILTER ====================
class AdaptiveFilter:
    """
    Median + EMA combo filter.
    More robust than simple moving average — rejects spike outliers.
    Matches Review-2 'noise filtering' requirement.
    """
    def __init__(self, n=3, window=7):
        self.wins = [deque(maxlen=window) for _ in range(n)]
        self.ema  = np.zeros(n)
        self.a    = 0.35
 
    def filter(self, x):
        out = np.zeros(len(x))
        for i, v in enumerate(x):
            self.wins[i].append(v)
            med = float(np.median(self.wins[i]))
            self.ema[i] = (1 - self.a) * self.ema[i] + self.a * med
            out[i] = self.ema[i]
        return out
 
 
# ==================== FATIGUE STATE MACHINE ====================
# ==================== FATIGUE STATE MACHINE ====================
# ==================== FATIGUE STATE MACHINE ====================
# ==================== FATIGUE STATE MACHINE (FINAL TUNED) ====================
class FatigueStateMachine:
    """
    FINAL VERSION — balanced for your hardware.
    - SAFE returns quickly when grip is released
    - CAUTION on normal/medium grip
    - WARNING on hard sustained grip
    - CRITICAL on sudden squeeze + release
    """
    STATE_IDLE      = "Idle"
    STATE_CONTACT   = "Contact Detection"
    STATE_ACQUIRE   = "Data Acquisition"
    STATE_BASELINE  = "Baseline Learning"
    STATE_MONITOR   = "Monitoring"
    STATE_STRESS    = "Stress Detected"
    STATE_ALERT     = "Alert"
    STATE_RECOVERY  = "Recovery"

    def __init__(self):
        self.score     = 0.0
        self.alert_idx = 0
        self.fsm_state = self.STATE_IDLE
        self.session_t = time.time()
        self.last_alert_t = 0
        self.n_samples = 0
        self.history   = deque(maxlen=200)

        # FINAL TUNED THRESHOLDS (this is the key change)
        self.UP_T   = [0, 26, 55, 82]    # harder to reach WARNING/CRITICAL
        self.DOWN_T = [0, 12, 35, 55]    # much easier to drop back to SAFE

    def update(self, pred_idx, confidence, mahal, tremor_ratio, gci, n_samples):
        self.n_samples = n_samples

        # FSM state transitions
        if self.fsm_state == self.STATE_IDLE and mahal > 0:
            self.fsm_state = self.STATE_CONTACT
        elif self.fsm_state == self.STATE_CONTACT:
            self.fsm_state = self.STATE_ACQUIRE
        elif self.fsm_state == self.STATE_ACQUIRE and n_samples > 5:
            self.fsm_state = self.STATE_BASELINE
        elif self.fsm_state == self.STATE_BASELINE and n_samples > 50:
            self.fsm_state = self.STATE_MONITOR

        if self.fsm_state not in (self.STATE_MONITOR, self.STATE_STRESS,
                                   self.STATE_ALERT,   self.STATE_RECOVERY):
            return 0, 0.0, self.fsm_state

        # Scoring
        anomaly_s = 45.0 if pred_idx == 3 else (30.0 if pred_idx == 2 else (18.0 if pred_idx == 1 else 0.0))
        mahal_s   = min(45.0, mahal * 18.0)
        tremor_s  = tremor_ratio * 35.0
        gci_s     = (1.0 - gci) * 20.0
        session_h = (time.time() - self.session_t) / 3600.0
        time_s    = min(12.0, session_h * 6.0)

        raw = anomaly_s + mahal_s + tremor_s + gci_s + time_s
        self.score = 0.75 * self.score + 0.25 * raw   # faster recovery when grip released
        self.history.append(self.score)

        # Alert transitions
        if self.score > self.UP_T[min(self.alert_idx + 1, 3)]:
            self.alert_idx = min(self.alert_idx + 1, 3)
        elif self.score < self.DOWN_T[self.alert_idx]:
            self.alert_idx = max(self.alert_idx - 1, 0)

        if self.alert_idx >= 2:
            self.fsm_state = self.STATE_STRESS if self.alert_idx == 2 else self.STATE_ALERT
        elif self.alert_idx == 1 and self.fsm_state in (self.STATE_STRESS, self.STATE_ALERT):
            self.fsm_state = self.STATE_RECOVERY
        elif self.alert_idx == 0:
            self.fsm_state = self.STATE_MONITOR

        return self.alert_idx, self.score, self.fsm_state

    def should_print_alert(self):
        now = time.time()
        if self.alert_idx >= 2 and (now - self.last_alert_t) > CONFIG["alert_cooldown"]:
            self.last_alert_t = now
            return True
        return False 
# ==================== DATA MANAGER ====================
class DataManager:
    def __init__(self):
        self.X   = []
        self.y   = []
        self.seq_buf = deque(maxlen=60)
        self.load()
 
    def load(self):
        if os.path.exists(CONFIG["data_path"]):
            with open(CONFIG["data_path"]) as f:
                for row in csv.reader(f):
                    try:
                        self.X.append([float(v) for v in row[:-1]])
                        self.y.append(int(row[-1]))
                    except: pass
            log.info(f"Loaded {len(self.X)} training samples")
 
    def add(self, x_norm, label):
        self.X.append(x_norm.tolist()); self.y.append(label)
 
    def save(self):
        with open(CONFIG["data_path"], "w", newline="") as f:
            w = csv.writer(f)
            for xi, yi in zip(self.X, self.y):
                w.writerow(xi + [yi])
 
    def get_sequence(self, x_norm, seq_len=10):
        self.seq_buf.append(x_norm)
        seq = list(self.seq_buf)
        while len(seq) < seq_len: seq = [seq[0]] + seq
        return np.array(seq[-seq_len:])
 
 
# ==================== TRAINER ====================
class Trainer:
    def __init__(self, model):
        self.model      = model
        self.opt        = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
        self.sched      = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=60)
        self.loss_fn    = nn.CrossEntropyLoss()
        self.last_n     = 0
        self._lock      = threading.Lock()
 
    def should_train(self, n):
        return n >= 20 and (n - self.last_n) >= CONFIG["retrain_every"]
 
    def train(self, data_mgr):
        with self._lock:
            X_raw = np.array(data_mgr.X)
            y_raw = np.array(data_mgr.y)
            if len(X_raw) < 20: return
 
            seqs, insts, trems, lbls = [], [], [], []
            for i in range(10, len(X_raw)):
                seq = X_raw[max(0,i-10):i]
                while len(seq) < 10: seq = np.vstack([seq[:1], seq])
                seqs.append(seq[-10:])
                insts.append(X_raw[i])
                trems.append(np.zeros(4))
                lbls.append(y_raw[i])
 
            if len(seqs) < 5: return
 
            Xs = torch.tensor(np.array(seqs),  dtype=torch.float32)
            Xi = torch.tensor(np.array(insts), dtype=torch.float32)
            Xt = torch.tensor(np.array(trems), dtype=torch.float32)
            y  = torch.tensor(np.array(lbls),  dtype=torch.long)
 
            self.model.train()
            for ep in range(120):
                logits, _ = self.model(Xs, Xi, Xt)
                loss = self.loss_fn(logits, y)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); self.sched.step()
                if loss.item() < 0.015: break
 
            self.model.eval()
            torch.save(self.model.state_dict(), CONFIG["model_path"])
            self.last_n = len(X_raw)
            log.info(f"✅ Model retrained | Loss={loss.item():.4f} | N={len(X_raw)}")
 
 
# ==================== MAIN ENGINE ====================
class NeuroGripEngine:
    def __init__(self):
        self.model   = NeuroGripNet(num_classes=CONFIG["num_classes"],
                                    seq_len=CONFIG["seq_len"])
        self.model.eval()
        self.baseline = PersonalizedBaseline(n_channels=3, alpha=0.04)
        self.tremor   = TremorAnalyzer(window=CONFIG["fft_window"],
                                       sr=CONFIG["sample_rate"])
        self.filt     = AdaptiveFilter(n=3)
        self.fsm      = FatigueStateMachine()
        self.data     = DataManager()
        self.trainer  = Trainer(self.model)
        self.ctx_win  = deque(maxlen=5)
        self.dyn_thr  = 0.50
        self.n        = 0
        self.session_events = []
        self._load()
 
    def _load(self):
        if os.path.exists(CONFIG["model_path"]):
            try:
                self.model.load_state_dict(
                    torch.load(CONFIG["model_path"], weights_only=True))
                log.info("✅ Loaded model")
            except Exception as e:
                log.warning(f"Model load failed: {e}")
        if os.path.exists(CONFIG["baseline_path"]):
            try:
                self.baseline.load(CONFIG["baseline_path"])
                log.info("✅ Loaded baseline")
            except: pass
 
    def process(self, raw_vals):
        """Full pipeline: filter → baseline → tremor → predict → FSM → alert."""
        x_raw = np.array(raw_vals, dtype=float)
 
        # 1. Adaptive noise filter (median + EMA)
        x_filt = self.filt.filter(x_raw)
 
        # 2. Online personalized baseline update
        self.baseline.update(x_filt)
        x_norm = self.baseline.normalize(x_filt)
 
        # 3. Tremor / frequency analysis → GCI
        self.tremor.update(x_norm)
        trem_feats = self.tremor.features()
        gci = float(trem_feats[3])
 
        # 4. Context smoothing (sequence awareness)
        self.ctx_win.append(x_norm)
        x_ctx = np.mean(self.ctx_win, axis=0)
 
        # 5. Sequence for LSTM
        x_seq = self.data.get_sequence(x_norm, seq_len=CONFIG["seq_len"])
 
        # 6. Prediction
        pred_idx, conf = self._predict(x_seq, x_ctx, trem_feats)
 
        # 7. Anomaly score
        mahal = self.baseline.mahalanobis(x_filt)
 
        # 8. Fatigue state machine
        al, score, fsm_state = self.fsm.update(
            pred_idx, conf, mahal, trem_feats[2], gci, self.n)
 
        self.n += 1
 
        # 9. Adaptive training (background thread)
        if self.trainer.should_train(len(self.data.X)):
            threading.Thread(target=self.trainer.train,
                             args=(self.data,), daemon=True).start()
 
        # 10. Periodic saves
        if self.n % 100 == 0:
            self.baseline.save(CONFIG["baseline_path"])
 
        return {
            "n":           self.n,
            "prediction":  LABELS[pred_idx],
            "pred_idx":    pred_idx,
            "confidence":  round(conf, 3),
            "alert_idx":   al,
            "alert":       ALERT_LVLS[al],
            "score":       round(score, 2),
            "mahal":       round(mahal, 3),
            "gci":         round(gci, 3),
            "tremor":      round(float(trem_feats[2]), 3),
            "dom_freq":    round(float(trem_feats[0]) * 10, 2),
            "fsm_state":   fsm_state,
            # Raw sensor values for display
            "fsr":         round(float(x_filt[0]), 4),
            "gsr":         round(float(x_filt[1]), 4),   # bio-impedance proxy
            "tremor_raw":  round(float(x_filt[2]), 4),
        }
 
    def _predict(self, x_seq, x_ctx, trem_feats):
        with torch.no_grad():
            s = torch.tensor(x_seq,      dtype=torch.float32).unsqueeze(0)
            i = torch.tensor(x_ctx,      dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(trem_feats, dtype=torch.float32).unsqueeze(0)
            logits, cal_conf = self.model(s, i, t)
            probs = F.softmax(logits, dim=1)
            pred  = int(torch.argmax(probs).item())
            conf  = float(torch.max(probs).item())
 
            # Dynamic threshold adaptation
            if conf > 0.82:
                self.dyn_thr = min(0.72, self.dyn_thr + 0.015)
            elif conf < 0.4:
                self.dyn_thr = max(0.38, self.dyn_thr - 0.012)
 
            final_conf = 0.7 * conf + 0.3 * float(cal_conf.item())
 
            if final_conf < self.dyn_thr:
                return 0, final_conf  # uncertain → default Relax
        return pred, final_conf
 
 
# ==================== SERIAL ====================
def connect_serial():
    try:
        ser = serial.Serial(CONFIG["port"], CONFIG["baud"], timeout=1)
        log.info(f"✅ ESP32 connected on {CONFIG['port']}")
        return ser
    except Exception as e:
        log.error(f"❌ Serial failed: {e}")
        return None
 
 
def send_to_esp32(ser, result):
    """Send ML decision back to ESP32 for OLED + buzzer output."""
    try:
        al = result['alert_idx']
        # Send ALERT command — ESP32 buzzer task reads this
        cmd = f"ALERT:{al}\n"
        ser.write(cmd.encode())
        ser.flush()  # FIX: flush immediately so ESP32 gets it without delay

        # Send display data every time
        state = result["prediction"][:10]
        disp  = f"DISP:{state}|{result['confidence']:.2f}|{result['score']:.1f}|{result['gci']:.2f}\n"
        ser.write(disp.encode())
        ser.flush()
    except Exception as e:
        log.warning(f"Serial write error: {e}")
 
 
# ==================== DISPLAY ====================
ALERT_BANNERS = {
    2: "⚠️  CAUTION — Driver stress elevated. Consider a break.",
    3: "🚨 CRITICAL — Pull over safely. Severe fatigue/stress detected."
}
 
def display_status(r):
    icon = ALERT_ICONS[r["alert_idx"]]
    print(
        f"\r[{r['n']:05d}] {icon} {r['alert']:8s} | "
        f"{r['prediction']:9s} | "
        f"Conf:{r['confidence']:.2f} | "
        f"Score:{r['score']:5.1f} | "
        f"GCI:{r['gci']:.2f} | "
        f"Tremor:{r['tremor']:.2f} | "
        f"GSR(Bio):{r['gsr']:.3f} | "
        f"FSR:{r['fsr']:.3f} | "
        f"FSM:{r['fsm_state']}",
        end="", flush=True
    )
    if r["alert_idx"] >= 2:
        print(f"\n{'!'*65}")
        print(f"  {ALERT_BANNERS.get(r['alert_idx'], '')}")
        print(f"  Fatigue Score: {r['score']:.1f}/100  |  "
              f"Mahal Dist: {r['mahal']:.2f}  |  "
              f"Dom Freq: {r['dom_freq']:.1f} Hz")
        print(f"{'!'*65}")
 
 
# ==================== SESSION LOGGER ====================
class SessionLogger:
    def __init__(self):
        self.events = []
        self.t0 = datetime.now().isoformat()
 
    def log(self, result):
        self.events.append({"t": datetime.now().isoformat(), **result})
 
    def save(self):
        with open(CONFIG["session_path"], "w") as f:
            json.dump({"start": self.t0, "end": datetime.now().isoformat(),
                       "events": self.events}, f, indent=2)
        log.info(f"Session saved: {len(self.events)} events")
 
 
# ==================== MAIN ====================
def run():
    ser = connect_serial()
    if not ser:
        log.error("No ESP32 connection — exiting."); return

    # FIX: Always clear stale baseline so re-learns from scratch
    # The old baseline was learned with bugs — delete and start fresh
    for stale in ["baseline.json"]:
        if os.path.exists(stale):
            os.remove(stale)
            log.info(f"Cleared stale {stale} — re-learning baseline")
 
    engine = NeuroGripEngine()
    slogger = SessionLogger()
    n = 0
 
    log.info("🚀 NeuroGrip Engine running. Collecting baseline for first 50 samples...\n")
 
    try:
        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if not line: continue
 
                # Skip heartbeat and status packets
                if line.startswith("HB") or line.startswith("NEUROGRIP"):
                    log.info(f"[ESP32] {line}"); continue
 
                parts = line.split()
                if len(parts) != 3: continue
 
                vals = [float(p) for p in parts]
                result = engine.process(vals)
                n += 1
 
                display_status(result)
                slogger.log(result)
 
                # FIX: Send ALERT every sample when alert>=1, else every 5
                # This ensures buzzer responds immediately
                if result["alert_idx"] >= 1 or n % 5 == 0:
                    send_to_esp32(ser, result)
 
            except ValueError:
                continue
            except Exception as e:
                log.error(f"Processing error: {e}")
 
    except KeyboardInterrupt:
        print("\n\nShutting down NeuroGrip...")
        engine.baseline.save(CONFIG["baseline_path"])
        engine.data.save()
        slogger.save()
        log.info("All data saved. Goodbye.")
 
 
if __name__ == "__main__":
    run()