"""
NeuroGrip — Data Collection & Labeling Tool  (FIXED v2.1)
==========================================================
Run this BEFORE neurogrip_main.py to collect labeled training data.
 
FIXES in this version:
  - ESP32 sends normalized float values (0.0–1.0), NOT raw ADC ints.
    Old script applied EMA on top → double-filtered → dead signal. REMOVED.
  - Skips ALL non-data lines: RAW:, HB, NEUROGRIP, QUEUE FAIL, etc.
  - Validates that parsed values are actually in 0–1 float range.
  - Shows live min/max so you can verify sensors are responding.
  - Saves after EVERY sample (not just at end) — no data loss on crash.
  - Backs up existing data_v2.csv before overwriting.
 
Usage:
    python collect_data.py
 
Workflow:
    1. Connect ESP32 via USB
    2. Run this script
    3. Follow prompts — simulate Relax / Stressed / Fatigued / Anomaly
    4. Script saves labeled data to data_v2.csv
    5. Then run neurogrip.py — model trains on your data
"""
 
import serial
import csv
import time
import os
import shutil
import numpy as np
from collections import deque
from datetime import datetime
 
# ──────────────────────────────────────────────
#  CONFIG — edit port if needed
# ──────────────────────────────────────────────
CONFIG = {
    "port":      "COM5",        # ← change to your port (Linux: /dev/ttyUSB0)
    "baud":      115200,
    "data_path": "data_v2.csv",
    "sample_rate": 20,          # Hz — must match ESP32 SAMPLE_RATE_HZ
}
 
LABELS = {
    0: "Relax      (rest hands normally on wheel — no pressure)",
    1: "Stressed   (grip tight, tense up, press hard)",
    2: "Fatigued   (loose, droopy, barely touching)",
    3: "Anomaly    (sudden squeeze then release, trembling)",
}
 
# ──────────────────────────────────────────────
#  CONNECT
# ──────────────────────────────────────────────
def connect():
    try:
        s = serial.Serial(CONFIG["port"], CONFIG["baud"], timeout=2)
        time.sleep(2)          # wait for ESP32 boot
        s.flushInput()
        print(f"✅ Connected to ESP32 on {CONFIG['port']}\n")
        return s
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        print("   Make sure:")
        print("   - ESP32 is plugged in")
        print("   - Port is correct (check Device Manager / dmesg)")
        print("   - No other program (Arduino Serial Monitor) has the port open")
        return None
 
 
# ──────────────────────────────────────────────
#  PARSE ONE SERIAL LINE
# ──────────────────────────────────────────────
def parse_line(line):
    """
    ESP32 sends processed normalized values: "0.1234 0.4879 0.0073"
    Returns np.array of 3 floats in [0,1], or None if line is garbage.
    """
    # Skip all known non-data lines
    if not line:
        return None
    skip_prefixes = ("HB", "NEUROGRIP", "RAW:", "QUEUE", "Read_us", "Proc_us", "#", "//")
    for pfx in skip_prefixes:
        if line.startswith(pfx):
            return None
 
    parts = line.split()
    if len(parts) != 3:
        return None
 
    try:
        vals = np.array([float(p) for p in parts])
    except ValueError:
        return None
 
    # Sanity check — ESP32 normalizes to [0,1]
    # If values are huge (>10) you're reading raw ADC — firmware mismatch!
    if np.any(vals > 10.0) or np.any(vals < -1.0):
        print(f"\n⚠️  WARNING: Got out-of-range values: {vals}")
        print("   Your ESP32 firmware may be sending RAW ADC instead of normalized.")
        print("   Check SAMPLE_RATE_HZ and the commsTask in your .ino file.")
        return None
 
    return vals
 
 
# ──────────────────────────────────────────────
#  COLLECT ONE SESSION
# ──────────────────────────────────────────────
def collect_session(ser, label_idx, duration_sec=15):
    """
    Collect `duration_sec` seconds of labeled sensor data.
    NO extra filtering here — ESP32 already EMA-filters.
    Returns list of [fsr, gsr, tremor] samples.
    """
    print(f"\n⏳ Collecting {duration_sec}s  →  Class [{label_idx}]  {LABELS[label_idx]}")
    print("   START NOW →", flush=True)
 
    samples = []
    mins = np.ones(3) * 9999
    maxs = np.zeros(3)
 
    t0 = time.time()
    timeout_warn = False
 
    while (time.time() - t0) < duration_sec:
        try:
            raw_line = ser.readline()
            if not raw_line:
                if not timeout_warn:
                    print("\n   ⚠️  No data from ESP32 — check connection")
                    timeout_warn = True
                continue
            timeout_warn = False
 
            line = raw_line.decode(errors="ignore").strip()
            vals = parse_line(line)
            if vals is None:
                continue
 
            samples.append(vals.tolist())
            mins = np.minimum(mins, vals)
            maxs = np.maximum(maxs, vals)
 
            remaining = int(duration_sec - (time.time() - t0))
            print(
                f"\r   FSR={vals[0]:.3f}  GSR={vals[1]:.3f}  Tremor={vals[2]:.3f}"
                f"  |  Samples:{len(samples):4d}  |  {remaining:2d}s left   ",
                end="", flush=True
            )
 
        except serial.SerialException as e:
            print(f"\n❌ Serial error: {e}")
            break
        except Exception:
            continue
 
    print(f"\n   ✅ Collected {len(samples)} samples for class {label_idx}")
    if len(samples) > 0:
        print(f"   Range — FSR:[{mins[0]:.3f}~{maxs[0]:.3f}]  "
              f"GSR:[{mins[1]:.3f}~{maxs[1]:.3f}]  "
              f"Tremor:[{mins[2]:.3f}~{maxs[2]:.3f}]")
 
        # Warn if FSR barely moved (sensor not responding)
        if (maxs[0] - mins[0]) < 0.01:
            print("   ⚠️  FSR barely changed! Check wiring on Pin 34.")
        if (maxs[1] - mins[1]) < 0.005:
            print("   ⚠️  GSR barely changed! Check wiring on Pin 35.")
    else:
        print("   ❌ No valid samples collected! Check ESP32 output.")
 
    return samples
 
 
# ──────────────────────────────────────────────
#  SAVE SAMPLES (append to CSV)
# ──────────────────────────────────────────────
def save_samples(samples, label_idx, path):
    if not samples:
        return
 
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        for s in samples:
            w.writerow(s + [label_idx])
 
    # Count current totals per class
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(path) as f:
        for row in csv.reader(f):
            try:
                counts[int(row[-1])] += 1
            except:
                pass
 
    print(f"   💾 Saved {len(samples)} samples (label={label_idx}) → {path}")
    print(f"   📊 Dataset totals: "
          f"Relax={counts[0]}  Stressed={counts[1]}  "
          f"Fatigued={counts[2]}  Anomaly={counts[3]}  "
          f"Total={sum(counts.values())}")
 
 
# ──────────────────────────────────────────────
#  VERIFY SENSOR (live stream for 5s)
# ──────────────────────────────────────────────
def verify_sensors(ser):
    """Quick 5-second live read so you can confirm sensors work before collecting."""
    print("\n🔍 Sensor Verification (5 seconds) — Move your hand on the FSR now...")
    t0 = time.time()
    count = 0
    while (time.time() - t0) < 5:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            vals = parse_line(line)
            if vals is None:
                continue
            count += 1
            print(f"\r   FSR={vals[0]:.4f}  GSR={vals[1]:.4f}  Tremor={vals[2]:.4f}  "
                  f"[{count} samples/5s]", end="", flush=True)
        except:
            continue
    print(f"\n   → Received {count} samples in 5 seconds (~{count//5} Hz)")
    if count < 10:
        print("   ⚠️  Very few samples! ESP32 may not be sending data correctly.")
        print("      Expected ~20 samples/sec. Check firmware SAMPLE_RATE_HZ.")
    else:
        print("   ✅ Sensors responding correctly!")
 
 
# ──────────────────────────────────────────────
#  BACKUP EXISTING DATA
# ──────────────────────────────────────────────
def backup_data(path):
    if os.path.exists(path):
        backup = path.replace(".csv", f"_backup_{datetime.now().strftime('%H%M%S')}.csv")
        shutil.copy(path, backup)
        print(f"   📁 Backed up existing data → {backup}")
 
 
# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    ser = connect()
    if not ser:
        return
 
    print("\n" + "=" * 60)
    print("  NeuroGrip Data Collection Tool  (Fixed v2.1)")
    print("  Collect labeled grip data from your hands")
    print("=" * 60)
 
    print("\nSensor channels (ESP32 sends normalized 0.0–1.0):")
    print("  Ch 0  FSR     Pin 34  — Grip pressure")
    print("  Ch 1  GSR     Pin 35  — Bio-impedance proxy (sweat/stress)")
    print("  Ch 2  Tremor  Pin 32  — Micro-movement proxy\n")
 
    print("Classes:")
    for k, v in LABELS.items():
        print(f"  [{k}] {v}")
 
    print("\n💡 Tips for good data:")
    print("   - Collect AT LEAST 3 rounds × 15s per class = ~900 samples/class")
    print("   - Both Janavi AND Hardik should collect separately")
    print("   - Actually change your grip for each class — don't fake it")
    print("   - Collect in the same environment you'll use the system\n")
 
    # Backup existing data
    backup_data(CONFIG["data_path"])
 
    # Sensor check
    print("\nPress ENTER to run a quick sensor verification, or 's' to skip: ", end="")
    choice = input().strip().lower()
    if choice != 's':
        verify_sensors(ser)
 
    total = 0
    try:
        while True:
            print("\n" + "-" * 50)
            print("Which class? (0/1/2/3) or 'q' to quit: ", end="")
            choice = input().strip()
            if choice.lower() == 'q':
                break
            try:
                label_idx = int(choice)
                if label_idx not in LABELS:
                    print("Invalid. Choose 0–3.")
                    continue
            except ValueError:
                print("Invalid input.")
                continue
 
            print(f"Duration in seconds? (default 15, min 10): ", end="")
            dur_str = input().strip()
            try:
                dur = max(10, int(dur_str)) if dur_str else 15
            except ValueError:
                dur = 15
 
            input(f"\nPress ENTER when ready → class [{label_idx}]  {LABELS[label_idx]}")
 
            # Flush stale serial buffer before collecting
            ser.flushInput()
 
            samples = collect_session(ser, label_idx, dur)
            if samples:
                save_samples(samples, label_idx, CONFIG["data_path"])
                total += len(samples)
                print(f"\n   Running total: {total} samples collected this session")
            else:
                print("   No samples saved. Try again.")
 
    except KeyboardInterrupt:
        pass
 
    print(f"\n\n✅ Data collection complete.")
    print(f"   Session total: {total} samples")
    print(f"   Saved to: {CONFIG['data_path']}")
    print(f"\n   Next step: run   python neurogrip.py")
    print(f"   The model will train on your collected data.\n")
 
    ser.close()
 
 
if __name__ == "__main__":
    main()