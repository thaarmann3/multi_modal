#!/usr/bin/env python3
"""
NeoPixel hue-sweep tester for your Arduino sketch.

Behavior:
- For each of 16 LEDs:
  - Sweep that LED through the full hue range over ~1 second (others off)
  - Move to the next LED
- Repeat forever

Serial format sent each frame:
  r,g,b;r,g,b;... (16 triples) + '\n'

Requires:
  pip install pyserial
"""

import sys
import time
import colorsys
import serial


NUM_PIXELS = 16
BAUD = 250000

# Tune these if you want
SWEEP_SECONDS_PER_PIXEL = 1.0
FPS = 60  # frames per second during the sweep


def hsv_to_rgb255(h: float, s: float = 1.0, v: float = 1.0):
    """h,s,v in [0,1] -> (r,g,b) in [0,255] ints."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r *20), int(g * 20), int(b * 20)


def build_line(active_index: int, rgb_active):
    """Build 'r,g,b;...'(16) with only one pixel lit."""
    parts = []
    for i in range(NUM_PIXELS):
        if i == active_index:
            r, g, b = rgb_active
        else:
            r, g, b = 0, 0, 0
        parts.append(f"{r},{g},{b}")
    return ";".join(parts) + "\n"


def main():

    port = "COM6"

    # Open serial. timeout=0 means non-blocking reads; we only write.
    with serial.Serial(port, BAUD, timeout=0) as ser:
        # Give Arduino time to reset after opening the port
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        frames_per_sweep = max(1, int(SWEEP_SECONDS_PER_PIXEL * FPS))
        frame_dt = SWEEP_SECONDS_PER_PIXEL / frames_per_sweep

        print(f"Streaming to {port} @ {BAUD} baud. Ctrl+C to stop.")

        try:
            while True:
                for pix in range(NUM_PIXELS):
                    t0 = time.perf_counter()
                    for f in range(frames_per_sweep):
                        # Hue goes 0..1 across the sweep
                        hue = f / frames_per_sweep
                        rgb = hsv_to_rgb255(hue, 1.0, 1.0)

                        line = build_line(pix, rgb)
                        ser.write(line.encode("ascii"))

                        # Pace to ~FPS (best-effort)
                        target = t0 + (f + 1) * frame_dt
                        now = time.perf_counter()
                        sleep_s = target - now
                        if sleep_s > 0:
                            time.sleep(sleep_s)

        except KeyboardInterrupt:
            # Turn all off on exit
            off_line = ";".join(["0,0,0"] * NUM_PIXELS) + "\n"
            ser.write(off_line.encode("ascii"))
            print("\nStopped. LEDs off.")


if __name__ == "__main__":
    main()
