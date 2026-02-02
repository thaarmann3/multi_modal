# Test script: toggle SHOW_PARTIALS / WITH_CONTROL_LOOP to try different subscriber modes. Ctrl+C to stop.
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stt import STTPublisher, STTConfig, TranscriptEvent

# Toggle: print interim partials as you speak.
SHOW_PARTIALS = False
# Toggle: run STT in background and a 100 Hz control loop (subscriber writes shared state, loop reads).
WITH_CONTROL_LOOP = False

CONTROL_RATE_HZ = 100
CONTROL_DT = 1.0 / CONTROL_RATE_HZ

# For WITH_CONTROL_LOOP: subscriber updates this; control_step() reads it.
latest_final_text = ""
latest_final_lock = threading.Lock()


def on_transcript(event: TranscriptEvent) -> None:
    if event.is_final and event.text.strip():
        text = event.text.strip().lower()
        if WITH_CONTROL_LOOP:
            with latest_final_lock:
                global latest_final_text
                latest_final_text = text
        print("[transcript] {}".format(text), flush=True)
    elif SHOW_PARTIALS and event.text.strip():
        print("  ... {}".format(event.text.strip()), end="\r", flush=True)


def control_step() -> None:
    # One tick of your control loop; use latest_final_text for voice commands.
    with latest_final_lock:
        text = latest_final_text
    if text:
        pass  # e.g. react to "stop", "go"
    pass  # your control logic


def main() -> None:
    pub = STTPublisher(STTConfig())
    pub.subscribe(on_transcript)

    if WITH_CONTROL_LOOP:
        pub.start_background()
        print("STT + control loop {} Hz. Speak to test; Ctrl+C to stop.".format(CONTROL_RATE_HZ), flush=True)
        next_heartbeat = time.perf_counter() + 1.0
        iterations = 0
        try:
            while True:
                t0 = time.perf_counter()
                control_step()
                iterations += 1
                if t0 >= next_heartbeat:
                    print("Control loop ({} it/s)".format(iterations), flush=True)
                    iterations = 0
                    next_heartbeat = t0 + 1.0
                elapsed = time.perf_counter() - t0
                time.sleep(max(0, CONTROL_DT - elapsed))
        except KeyboardInterrupt:
            pub.stop()
    else:
        print("Listening. Ctrl+C to stop.", flush=True)
        try:
            pub.start()
        except KeyboardInterrupt:
            pub.stop()


if __name__ == "__main__":
    main()
