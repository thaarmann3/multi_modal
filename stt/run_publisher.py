# CLI: run publisher and print every transcript (partial + final). Ctrl+C to stop.
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stt import STTPublisher, STTConfig, TranscriptEvent


def on_event(event: TranscriptEvent) -> None:
    kind = "FINAL" if event.is_final else "partial"
    print(f"[{kind}] {event.text}", flush=True)


if __name__ == "__main__":
    pub = STTPublisher(STTConfig())
    pub.subscribe(on_event)
    print("Listening. Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        pub.start()
    except KeyboardInterrupt:
        pub.stop()
