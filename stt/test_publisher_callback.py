import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stt import STTPublisher, STTConfig


class _FakeStatus:
    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return "input overflow"


class _BrokenStderr:
    def __init__(self) -> None:
        self.write_calls = 0

    def write(self, message: str) -> int:
        self.write_calls += 1
        raise OSError(6, "The handle is invalid")

    def flush(self) -> None:
        pass


class TestPublisherAudioCallback(unittest.TestCase):
    def test_audio_callback_ignores_invalid_stderr_handle(self) -> None:
        pub = STTPublisher(STTConfig())
        broken_stderr = _BrokenStderr()

        with patch.object(sys, "stderr", broken_stderr):
            pub._audio_callback(b"\x01\x02", 1, None, _FakeStatus())
            pub._audio_callback(b"\x03\x04", 1, None, _FakeStatus())

        self.assertEqual(pub._audio_queue.get_nowait(), b"\x01\x02")
        self.assertEqual(pub._audio_queue.get_nowait(), b"\x03\x04")
        self.assertEqual(broken_stderr.write_calls, 1)


if __name__ == "__main__":
    unittest.main()
