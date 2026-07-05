"""
BUJJI — Clap Calibration Tool
==============================
Run this BEFORE running main.py to find the right CLAP_THRESHOLD for your mic.

Steps:
  1. Run: python calibrate_clap.py
  2. Stay SILENT for the first 5 seconds  → see your ambient noise level
  3. Clap normally a few times            → see your clap peak levels
  4. Use the suggested threshold printed at the end
"""

import pyaudio
import numpy as np
import time

CHUNK       = 1024
SAMPLE_RATE = 16000
DURATION    = 15   # seconds to monitor

pa     = pyaudio.PyAudio()

# ── Print available input devices ──────────────────────────
print("\n─── Available Microphones ───────────────────────────────")
default_index = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        is_default = ""
        if pa.get_default_input_device_info()["index"] == i:
            is_default = "  ← DEFAULT"
            default_index = i
        print(f"  [{i}] {info['name']}{is_default}")

print(f"\nUsing device index: {default_index}")
print("─────────────────────────────────────────────────────────\n")

try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=default_index,
        frames_per_buffer=CHUNK,
    )
except Exception as e:
    print(f"[ERROR] Could not open mic: {e}")
    pa.terminate()
    exit(1)

print(f"Monitoring mic for {DURATION} seconds...")
print("→ Stay SILENT for first 5s, then CLAP a few times\n")
print(f"{'Time':>5}  {'Peak':>7}  Bar")
print("─" * 50)

peaks      = []
start_time = time.time()

try:
    while time.time() - start_time < DURATION:
        data    = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
        peak    = int(np.abs(samples).max())
        peaks.append(peak)

        elapsed = time.time() - start_time
        bar     = "█" * min(50, peak // 200)
        print(f"{elapsed:>4.1f}s  {peak:>7}  {bar}")

except KeyboardInterrupt:
    print("\n[Stopped early]")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

# ── Summary ────────────────────────────────────────────────
if peaks:
    ambient = min(peaks[:30]) if len(peaks) >= 30 else min(peaks)
    max_peak = max(peaks)
    suggested = int(ambient * 5 + (max_peak - ambient) * 0.3)
    suggested = max(suggested, ambient * 8)   # at least 8x ambient noise

    print("\n─── Calibration Summary ──────────────────────────────────")
    print(f"  Ambient noise level : {ambient}")
    print(f"  Loudest clap peak   : {max_peak}")
    print(f"  Suggested threshold : {suggested}")
    print("\n  In main.py, change line:")
    print(f"    CLAP_THRESHOLD = 3000")
    print(f"  To:")
    print(f"    CLAP_THRESHOLD = {suggested}")
    print("─────────────────────────────────────────────────────────\n")