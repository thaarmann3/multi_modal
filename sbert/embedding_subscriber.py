"""
STT → SBERT pipeline: subscribe to STT transcriptions, encode with SBERT, output embeddings.
Prints each sentence for review and optionally invokes a callback with (sentence, embedding).

Model loading:
- STT (Vosk): uses local model in stt/models/; no download after first setup.
- SBERT: uses a project-local cache (default sbert/.models/). First run downloads;
  subsequent runs load from cache. Pass sbert_cache_dir to override.
- To avoid reloading models in memory: keep one process running, or reuse
  EmbeddingPipeline in the same process (it holds loaded models).
"""
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stt import STTPublisher, STTConfig, TranscriptEvent


# Default model used in sbert/test.py
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Project-local cache: first run downloads here; later runs load from here (no re-download).
DEFAULT_SBERT_CACHE_DIR = Path(__file__).resolve().parent / ".models"


def _load_sbert_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_folder: Optional[Union[Path, str]] = None,
):
    import sentence_transformers

    cache = cache_folder if cache_folder is not None else DEFAULT_SBERT_CACHE_DIR
    cache = Path(cache) if isinstance(cache, str) else cache
    cache.mkdir(parents=True, exist_ok=True)
    return sentence_transformers.SentenceTransformer(model_name, cache_folder=str(cache))


class EmbeddingPipeline:
    """
    Holds STT publisher and SBERT model; loads once and reuses for multiple runs
    in the same process. Use this when you want to start/stop listening several
    times without reloading models.
    """

    def __init__(
        self,
        stt_config: Optional[STTConfig] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        sbert_cache_dir: Optional[Union[Path, str]] = None,
    ):
        self.stt_config = stt_config or STTConfig()
        self.model_name = model_name
        self.sbert_cache_dir = sbert_cache_dir
        self._model = None
        self._pub: Optional[STTPublisher] = None
        self._subscriber: Optional[Callable[[TranscriptEvent], None]] = None

    @property
    def model(self):
        """Load SBERT model once; reuse thereafter."""
        if self._model is None:
            self._model = _load_sbert_model(self.model_name, self.sbert_cache_dir)
        return self._model

    @property
    def pub(self) -> STTPublisher:
        """Create STT publisher once; reuse thereafter."""
        if self._pub is None:
            self._pub = STTPublisher(self.stt_config)
        return self._pub

    def run(
        self,
        embedding_callback: Optional[Callable[[str, np.ndarray], None]] = None,
        print_embedding_shape: bool = True,
    ) -> None:
        """Run STT → SBERT (blocking). Uses cached model/publisher. Ctrl+C to stop."""
        model = self.model
        pub = self.pub

        def on_transcript(event: TranscriptEvent) -> None:
            if not event.is_final or not event.text.strip():
                return
            sentence = event.text.strip()
            embedding = model.encode([sentence], convert_to_numpy=True)[0]
            print("[sentence] {}".format(sentence), flush=True)
            if print_embedding_shape:
                print("  embedding shape: {}".format(embedding.shape), flush=True)
            if embedding_callback:
                try:
                    embedding_callback(sentence, embedding)
                except Exception:
                    pass

        if self._subscriber is not None:
            pub.unsubscribe(self._subscriber)
        self._subscriber = on_transcript
        pub.subscribe(on_transcript)
        print(
            "STT → SBERT ({}). Speak to get embeddings; Ctrl+C to stop.".format(
                self.model_name
            ),
            flush=True,
        )
        try:
            pub.start()
        except KeyboardInterrupt:
            pub.stop()


def run_embedding_subscriber(
    stt_config: Optional[STTConfig] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    sbert_cache_dir: Optional[Union[Path, str]] = None,
    embedding_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    print_embedding_shape: bool = True,
) -> None:
    """
    Run STT publisher and, for each final transcription, encode with SBERT and output.

    - STT model is loaded from stt config (local path; no download).
    - SBERT model is cached in sbert_cache_dir (default sbert/.models/). First run
      downloads; later runs load from cache.
    - Prints each sentence for review.
    - Optionally calls embedding_callback(sentence, embedding) for downstream use.
    - If print_embedding_shape is True, prints embedding shape after each sentence.

    To avoid reloading models in memory across multiple "sessions", use
    EmbeddingPipeline once and call .run() each time.
    Ctrl+C to stop.
    """
    model = _load_sbert_model(model_name, sbert_cache_dir)
    pub = STTPublisher(stt_config or STTConfig())

    def on_transcript(event: TranscriptEvent) -> None:
        if not event.is_final or not event.text.strip():
            return
        sentence = event.text.strip()
        embedding = model.encode([sentence], convert_to_numpy=True)[0]
        print("[sentence] {}".format(sentence), flush=True)
        if print_embedding_shape:
            print("  embedding shape: {}".format(embedding.shape), flush=True)
        if embedding_callback:
            try:
                embedding_callback(sentence, embedding)
            except Exception:
                pass

    pub.subscribe(on_transcript)
    print(
        "STT → SBERT ({}). Speak to get embeddings; Ctrl+C to stop.".format(model_name),
        flush=True,
    )
    try:
        pub.start()
    except KeyboardInterrupt:
        pub.stop()


def main() -> None:
    run_embedding_subscriber()


if __name__ == "__main__":
    main()
