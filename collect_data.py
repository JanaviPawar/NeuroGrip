import argparse
import csv
import os
import random
import time
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports

DATA_PATH = "data_v3.csv"
HEADER = ["fsr", "gsr", "tremor", "label", "session", "person"]

CLASSES = {
    0: ("Relax", "Rest both hands on the wheel normally. Light, steady, comfortable grip."),
    1: ("Stressed", "Grip HARD and hold it. Mix it up between rounds: some moderate-tight, "
                    "some as hard as you can sustain. Keep the hands still."),
    2: ("Fatigued", "Loose, heavy hands. Let the grip slowly sag and drift, nearly letting go "
                    "now and then, like you're fighting sleep."),
    3: ("Anomaly", "Sudden events: quick hard squeeze then release, jerky grabs, and shake "
                   "the wheel with small trembling movements."),
}


def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        d = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in d for k in ("cp210", "ch340", "ch910", "usb-serial", "usb serial", "uart", "esp32", "silicon labs", "wch")):
            return p.device
    return ports[0].device if len(ports) == 1 else None


def connect(port):
    port = port or find_port()
    if not port:
        print("Couldn't find the ESP32. Plug it in, then run with --port COMx")
        print("Ports seen:", [p.device for p in serial.tools.list_ports.comports()] or "none")
        return None
    try:
        s = serial.Serial(port, 115200, timeout=2)
        time.sleep(2)
        s.reset_input_buffer()
        print(f"Connected to ESP32 on {port}")
        return s
    except Exception as e:
        print(f"Can't open {port}: {e}")
        print("Close the Arduino Serial Monitor (and neurogrip.py) if they're open, then retry.")
        return None


def parse_line(line):
    parts = line.split()
    if len(parts) != 3:
        return None             # skips HB, NEUROGRIP_READY, MPU_OK, etc.
    try:
        v = np.array([float(p) for p in parts])
    except ValueError:
        return None
    if np.any(v > 10) or np.any(v < -1):
        return None
    return v


def stream(ser, seconds, label=None):
    ser.reset_input_buffer()
    samples, t0 = [], time.time()
    while time.time() - t0 < seconds:
        raw = ser.readline()
        if not raw:
            print("\n  no data from ESP32, check the cable")
            continue
        v = parse_line(raw.decode(errors="ignore").strip())
        if v is None:
            continue
        samples.append(v.tolist())
        left = int(seconds - (time.time() - t0))
        print(f"\r  FSR={v[0]:.3f}  GSR={v[1]:.3f}  Tremor={v[2]:.3f}  | {len(samples):4d} samples | {left:2d}s left ",
              end="", flush=True)
    print()
    return np.array(samples)


def verify(ser):
    print("\nSensor check (8 s): squeeze the FSR, turn the GSR knob, and tap/shake the wheel.")
    s = stream(ser, 8)
    if len(s) < 50:
        print(f"  Only {len(s)} samples in 8 s (expected ~160). Check the firmware is running.")
        return
    rng = s.max(0) - s.min(0)
    print(f"  {len(s)/8:.0f} samples/s   range FSR {rng[0]:.3f}  GSR {rng[1]:.3f}  Tremor {rng[2]:.3f}")
    if rng[0] < 0.05: print("  FSR barely moved: check GPIO34 wiring.")
    if rng[1] < 0.02: print("  GSR barely moved: check the pot on GPIO35.")
    if rng[2] < 0.05: print("  Tremor barely moved: is the MPU6050 working? (Arduino Serial Monitor should say MPU_OK at boot.)")
    if s[:, 2].min() > 0.3: print("  Tremor never drops low even when still: raise TREMOR_ACC_FS_G / TREMOR_GYRO_FS_DPS in the .ino.")


def save(samples, label, session, person):
    new = not os.path.exists(DATA_PATH)
    with open(DATA_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for s in samples:
            w.writerow([f"{s[0]:.4f}", f"{s[1]:.4f}", f"{s[2]:.4f}", label, session, person])
    counts, sessions = [0] * 4, [set() for _ in range(4)]
    with open(DATA_PATH, newline="") as f:
        for r in csv.reader(f):
            try:
                c = int(r[3]); counts[c] += 1; sessions[c].add(r[4])
            except (ValueError, IndexError):
                pass
    print("  Dataset now: " + "  ".join(f"{CLASSES[i][0]} {counts[i]} ({len(sessions[i])} rounds)" for i in range(4)))


def record_round(ser, label, person, seconds):
    name, how = CLASSES[label]
    print(f"\n--- [{label}] {name} ---")
    print(f"  {how}")
    print("  Set the GSR knob to ANY position (don't match it to the class).")
    input("  Get into position, then press ENTER to start ...")
    for c in (3, 2, 1):
        print(f"  {c}...", end=" ", flush=True); time.sleep(0.7)
    print("GO")
    s = stream(ser, seconds, label)
    if len(s) < seconds * 10:
        print(f"  Only {len(s)} samples: round discarded.")
        return False
    ans = input("  Keep this round? (ENTER = yes, n = discard): ").strip().lower()
    if ans == "n":
        print("  Discarded.")
        return False
    session = f"{person}-{datetime.now().strftime('%m%d-%H%M%S')}-c{label}"
    save(s, label, session, person)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="who is holding the wheel, e.g. janavi / hardik / akhil")
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3, help="guided mode: rounds per class in this sitting")
    ap.add_argument("--manual", action="store_true")
    a = ap.parse_args()
    person = a.person.strip().lower().replace(",", "")

    ser = connect(a.port)
    if not ser:
        return
    if input("Run the sensor check first? (ENTER = yes, s = skip): ").strip().lower() != "s":
        verify(ser)

    try:
        if a.manual:
            while True:
                c = input("\nClass 0 Relax / 1 Stressed / 2 Fatigued / 3 Anomaly, or q to quit: ").strip()
                if c.lower() == "q":
                    break
                if c in ("0", "1", "2", "3"):
                    record_round(ser, int(c), person, a.seconds)
        else:
            plan = [c for c in range(4) for _ in range(a.rounds)]
            random.shuffle(plan)
            print(f"\nGuided session: {len(plan)} rounds of {a.seconds}s in shuffled order. Ctrl+C stops any time; finished rounds are already saved.")
            for i, c in enumerate(plan, 1):
                print(f"\nRound {i}/{len(plan)}")
                while not record_round(ser, c, person, a.seconds):
                    if input("  Redo this round? (ENTER = yes, s = skip): ").strip().lower() == "s":
                        break
    except KeyboardInterrupt:
        print("\nStopped.")
    ser.close()
    print(f"\nSaved to {DATA_PATH}. Next: python train_eval.py")


if __name__ == "__main__":
    main()