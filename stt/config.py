# STT config: where the model lives and how audio is captured.
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Vosk model dir; override via STTConfig(model_path=...).
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"


@dataclass
class STTConfig:
    model_path: Path = DEFAULT_MODEL_DIR
    sample_rate: int = 16000  # Vosk small model expects 16k
    block_ms: int = 100  # Smaller = lower latency, more CPU
    device: Optional[int] = None  # sounddevice input device index; None = default mic
    enable_agc: bool = False  # Toggle AGC on/off

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
