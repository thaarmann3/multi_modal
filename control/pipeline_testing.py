import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sbert.embedding_to_pf_subscriber import EmbeddingToPFPipeline

if __name__ == "__main__":
    pipeline = EmbeddingToPFPipeline()
    _ = pipeline.model  # load SBERT (and PF model on first use) before starting
    pipeline.start_background(use_nn=True, print_params=False)
    time.sleep(0.5)  # let STT thread start and load Vosk before control loop

    try:
        while True:
            print("robot command", flush=True)
            sentence, emb, ridge, nn = pipeline.get_latest()
            if nn is not None:
                print("nn:", nn, flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pipeline.stop()
