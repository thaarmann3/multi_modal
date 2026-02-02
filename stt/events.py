# Payload sent to subscribers: one chunk of recognized text.
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool  # True when utterance ended; False for interim partials
    timestamp: float
