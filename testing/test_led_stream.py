#!/usr/bin/env python3
"""
Standalone test for the LED part of impedance_control_discretized_tunneling_LEDs.py.
Run this without the robot to verify:
  - Serial port opens (correct port/baud in config)
  - Arduino receives and displays frames (16 RGB triples: r,g,b;r,g,b;...\\n)
  - pyserial is installed

Usage: python test_led_stream.py [--duration SEC]
Default: runs a 10 s pattern (red -> green -> rotating), then exits.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import serial
except ImportError:
    print("[LED test] pyserial not installed. Install with: pip install pyserial")
    sys.exit(1)

# Load same config as main script
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "configs", "impedance_control_discretized_tunneling.json")
with open(config_path) as f:
    config = json.load(f)

LED_COUNT = config["leds"]["count"]
ARDUINO_PORT = config["leds"]["port"]
ARDUINO_BAUD = config["leds"]["baud"]


def frame_to_line(rgb_list):
    """Same format as ArduinoLedStreamer: 16 x 'r,g,b' joined by ';' + newline."""
    parts = [f"{int(r)},{int(g)},{int(b)}" for (r, g, b) in rgb_list]
    return ";".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Test LED streaming (no robot)")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run pattern")
    args = parser.parse_args()

    if not ARDUINO_PORT:
        print("[LED test] No LED port in config (leds.port). Set it and try again.")
        sys.exit(1)

    print(f"[LED test] Opening {ARDUINO_PORT} @ {ARDUINO_BAUD} baud, {LED_COUNT} LEDs...")
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0, write_timeout=1.0)
        time.sleep(2.0)  # let Arduino reset after open
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"[LED test] Failed to open serial: {e}")
        print("  - Check port name (e.g. COM6 on Windows, /dev/ttyACM0 on Linux)")
        print("  - Check Arduino is connected and not used by another program")
        sys.exit(1)

    print("[LED test] Sending patterns. You should see LEDs change (red -> green -> rotating).")
    start = time.monotonic()
    step = 0
    try:
        while (time.monotonic() - start) < args.duration:
            t = time.monotonic() - start
            if t < 3:
                # All red
                rgb_list = [(30, 0, 0)] * LED_COUNT
            elif t < 6:
                # All green
                rgb_list = [(0, 30, 0)] * LED_COUNT
            else:
                # Rotating single LED (white)
                i = int(t * 4) % LED_COUNT
                rgb_list = [(0, 0, 0)] * LED_COUNT
                rgb_list[i] = (25, 25, 25)

            line = frame_to_line(rgb_list)
            ser.write(line.encode("ascii"))
            time.sleep(1.0 / 30)  # ~30 Hz
    finally:
        # Turn off
        off = frame_to_line([(0, 0, 0)] * LED_COUNT)
        try:
            ser.write(off.encode("ascii"))
        except Exception:
            pass
        ser.close()

    print("[LED test] Done. If you saw red -> green -> rotating LED, the LED path is working.")


if __name__ == "__main__":
    main()
