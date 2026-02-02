"""
STT → SBERT → PF params. Exported MATLAB model from sbert/prediction_models/.
Control loop: pipeline = EmbeddingToPFPipeline(); pipeline.start_background();
then each tick: sentence, emb, ridge, nn = pipeline.get_latest(); use nn/ridge as needed; pipeline.stop() when done.
"""
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stt import STTPublisher, STTConfig, TranscriptEvent
from sbert.embedding_subscriber import DEFAULT_MODEL_NAME, _load_sbert_model

_PF_MAT = Path(__file__).resolve().parent / "prediction_models" / "exported_models_like_cv.mat"

# MATLAB transfer functions
def _tansig(x): return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0
def _logsig(x): return 1.0 / (1.0 + np.exp(-x))
def _purelin(x): return x
def _poslin(x): return np.maximum(0.0, x)
_ACT = {"tansig": _tansig, "logsig": _logsig, "purelin": _purelin, "poslin": _poslin}


def _ensure_1d(a): return np.asarray(a).reshape(-1)


def _to_str_list(x):
    if isinstance(x, str): return [x]
    x = np.atleast_1d(x)
    out = []
    for s in x:
        if isinstance(s, str): out.append(s)
        elif isinstance(s, bytes): out.append(s.decode("utf-8"))
        elif isinstance(s, np.ndarray):
            out.append("".join(s.tolist()).strip() if s.dtype.kind in ("U", "S") else "".join(chr(int(c)) for c in s.flatten()).strip())
        else: out.append(str(s))
    return out


def _get_scalar_str(v):
    if isinstance(v, (bytes, str)): return v.decode("utf-8") if isinstance(v, bytes) else v
    if isinstance(v, np.ndarray): lst = _to_str_list(v); return lst[0] if lst else ""
    return str(v)


def _reshape_W3(W3, H2):
    W3 = np.asarray(W3, dtype=np.float64)
    if W3.ndim == 3:
        if W3.shape[0] == 1 and W3.shape[1] == H2 and W3.shape[2] == 4: return W3
        if W3.shape[0] == H2 and W3.shape[1] == 1 and W3.shape[2] == 4: return np.transpose(W3, (1, 0, 2))
        if W3.shape[0] == 4 and W3.shape[1] == H2 and W3.shape[2] == 1: return np.transpose(W3, (2, 1, 0))
        W3 = W3.reshape(-1)
    if W3.ndim == 2:
        if W3.shape == (H2, 4): return W3[np.newaxis, :, :]
        if W3.shape == (4, H2): return W3.T[np.newaxis, :, :]
        W3 = W3.reshape(-1)
    flat = W3.reshape(-1)
    if flat.size != H2 * 4: raise ValueError(f"W3 size {flat.size} != {H2*4}")
    return flat.reshape(1, H2, 4, order="F")


def _reshape_W(W, n1, n2):
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 3:
        if W.shape[0] == n1 and W.shape[1] == n2 and W.shape[2] == 4: return W
        if W.shape[0] == 4 and W.shape[1] == n1 and W.shape[2] == n2: return np.transpose(W, (1, 2, 0))
        if W.shape[0] == n2 and W.shape[1] == n1 and W.shape[2] == 4: return np.transpose(W, (1, 0, 2))
        W = W.reshape(-1)
    if W.ndim == 2 and W.shape == (n1, n2): raise ValueError(f"W single-output shape {W.shape}")
    flat = W.reshape(-1)
    if flat.size != n1 * n2 * 4: raise ValueError(f"W size {flat.size} != {n1*n2*4}")
    return flat.reshape(n1, n2, 4, order="F")


def _reshape_b(b, H):
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 2:
        if b.shape == (H, 4): return b
        if b.shape == (4, H): return b.T
        b = b.reshape(-1)
    flat = b.reshape(-1)
    if flat.size != H * 4: raise ValueError(f"b size {flat.size} != {H*4}")
    return flat.reshape(H, 4, order="F")


def _load_pf_model():
    if _load_pf_model._cached is not None:
        return _load_pf_model._cached
    mat = loadmat(_PF_MAT, squeeze_me=True, struct_as_record=False)
    if _get_scalar_str(mat.get("nnBackend_char", "feedforwardnet")) != "feedforwardnet":
        raise RuntimeError("Expected feedforwardnet export")
    muX = _ensure_1d(mat["muX"]).astype(np.float64)
    sigX = _ensure_1d(mat["sigX"]).astype(np.float64)
    muY = _ensure_1d(mat["muY"]).astype(np.float64)
    sigY = _ensure_1d(mat["sigY"]).astype(np.float64)
    B = np.asarray(mat["B_ridge"], dtype=np.float64)
    yNames = _to_str_list(mat.get("yNames_cell", ["x_m", "y_m", "amplitude", "radius"]))
    tf1, tf2, tf3 = _to_str_list(mat["tf1_cell"]), _to_str_list(mat["tf2_cell"]), _to_str_list(mat["tf3_cell"])
    p = muX.shape[0]
    H1 = int(np.asarray(mat["b1"], dtype=np.float64).size // 4)
    H2 = int(np.asarray(mat["b2"], dtype=np.float64).size // 4)
    W1 = _reshape_W(mat["W1"], H1, p)
    b1 = _reshape_b(mat["b1"], H1)
    W2 = _reshape_W(mat["W2"], H2, H1)
    b2 = _reshape_b(mat["b2"], H2)
    W3 = _reshape_W3(mat["W3"], H2)
    b3 = np.asarray(mat["b3"], dtype=np.float64).reshape(-1)
    if b3.size != 4: raise ValueError(f"b3 size {b3.size} != 4")
    b3 = b3.reshape(1, 4)
    _load_pf_model._cached = {
        "muX": muX, "sigX": sigX, "muY": muY, "sigY": sigY, "B": B, "yNames": yNames,
        "tf1": tf1, "tf2": tf2, "tf3": tf3,
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "p": p,
    }
    return _load_pf_model._cached


_load_pf_model._cached = None


def _ridge(xz, m):
    return m["muY"] + m["sigY"] * (np.concatenate([np.ones(1), xz]) @ m["B"])


def _nn(xz, m):
    y = np.empty(4, dtype=np.float64)
    for j in range(4):
        a1 = _ACT[m["tf1"][j]](m["W1"][:, :, j] @ xz + m["b1"][:, j])
        a2 = _ACT[m["tf2"][j]](m["W2"][:, :, j] @ a1 + m["b2"][:, j])
        yz = _ACT[m["tf3"][j]](m["W3"][0, :, j] @ a2 + m["b3"][0, j])
        y[j] = m["muY"][j] + m["sigY"][j] * yz
    return y


def predict_from_embedding(x: np.ndarray, use_ridge: bool = False, use_nn: bool = True) -> tuple[Optional[dict], Optional[dict]]:
    """PF params from SBERT embedding. Returns (ridge_params, nn_params); disabled predictor returns None."""
    m = _load_pf_model()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != m["p"]:
        raise ValueError(f"Embedding dim {x.shape[0]} != model dim {m['p']}")
    xz = (x - m["muX"]) / m["sigX"]
    r = dict(zip(m["yNames"], _ridge(xz, m))) if use_ridge else None
    n = dict(zip(m["yNames"], _nn(xz, m))) if use_nn else None
    return r, n


def _process_final(sentence: str, model, print_params: bool, cb: Optional[Callable],
                   use_ridge: bool, use_nn: bool) -> None:
    emb = model.encode([sentence], convert_to_numpy=True)[0]
    ridge, nn = predict_from_embedding(emb, use_ridge=use_ridge, use_nn=use_nn)
    if print_params:
        print("[sentence]", sentence, flush=True)
        if ridge is not None: print("  ridge:", ridge, flush=True)
        if nn is not None: print("  nn:", nn, flush=True)
    if cb:
        try: cb(sentence, emb, ridge, nn)
        except Exception: pass


def _make_on_transcript(model, print_params: bool, cb: Optional[Callable], use_ridge: bool, use_nn: bool):
    def on_transcript(event: TranscriptEvent) -> None:
        if not event.is_final or not event.text.strip():
            return
        _process_final(event.text.strip(), model, print_params, cb, use_ridge, use_nn)
    return on_transcript


class EmbeddingToPFPipeline:
    """STT → SBERT → PF params. Holds STT + SBERT; loads once, reuses. Use start_background() + get_latest() in a control loop."""

    def __init__(self, stt_config: Optional[STTConfig] = None, model_name: str = DEFAULT_MODEL_NAME,
                 sbert_cache_dir: Optional[Union[Path, str]] = None):
        self.stt_config = stt_config or STTConfig()
        self.model_name = model_name
        self.sbert_cache_dir = sbert_cache_dir
        self._model = self._pub = self._subscriber = None
        self._latest_sentence: Optional[str] = None
        self._latest_embedding: Optional[np.ndarray] = None
        self._latest_ridge: Optional[dict] = None
        self._latest_nn: Optional[dict] = None
        self._latest_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None: self._model = _load_sbert_model(self.model_name, self.sbert_cache_dir)
        return self._model

    @property
    def pub(self) -> STTPublisher:
        if self._pub is None: self._pub = STTPublisher(self.stt_config)
        return self._pub

    def _on_final_background(self, sentence: str, emb: np.ndarray, ridge: Optional[dict], nn: Optional[dict],
                             print_params: bool) -> None:
        with self._latest_lock:
            self._latest_sentence = sentence
            self._latest_embedding = emb.copy()
            self._latest_ridge = dict(ridge) if ridge is not None else None
            self._latest_nn = dict(nn) if nn is not None else None
        if print_params:
            print("[voice]", sentence, flush=True)
            if ridge is not None: print("  ridge:", ridge, flush=True)
            if nn is not None: print("  nn:", nn, flush=True)

    def start_background(self, use_ridge: bool = False, use_nn: bool = True, print_params: bool = True) -> None:
        """Start STT + SBERT + PF in a background thread. Params written on each final utterance; read them with get_latest()."""
        def on_transcript(event: TranscriptEvent) -> None:
            if not event.is_final or not event.text.strip():
                return
            sentence = event.text.strip()
            emb = self.model.encode([sentence], convert_to_numpy=True)[0]
            ridge, nn = predict_from_embedding(emb, use_ridge=use_ridge, use_nn=use_nn)
            self._on_final_background(sentence, emb, ridge, nn, print_params)
        if self._subscriber is not None:
            self.pub.unsubscribe(self._subscriber)
        self._subscriber = on_transcript
        self.pub.subscribe(on_transcript)
        self.pub.start_background()

    def get_latest(self) -> Tuple[Optional[str], Optional[np.ndarray], Optional[dict], Optional[dict]]:
        """Return (sentence, embedding, ridge_params, nn_params) from the last voice detection, then clear. One-time use per utterance."""
        with self._latest_lock:
            s = self._latest_sentence
            e = self._latest_embedding.copy() if self._latest_embedding is not None else None
            r = dict(self._latest_ridge) if self._latest_ridge is not None else None
            n = dict(self._latest_nn) if self._latest_nn is not None else None
            self._latest_sentence = None
            self._latest_embedding = None
            self._latest_ridge = None
            self._latest_nn = None
        return s, e, r, n

    def stop(self) -> None:
        """Stop the background STT."""
        self.pub.stop()

    def run(self, params_callback: Optional[Callable[[str, np.ndarray, Optional[dict], Optional[dict]], None]] = None,
            print_params: bool = True, use_ridge: bool = False, use_nn: bool = True) -> None:
        """Blocking: STT → SBERT → PF until Ctrl+C. For background use, call start_background() then get_latest() in your loop."""
        sub = _make_on_transcript(self.model, print_params, params_callback, use_ridge, use_nn)
        if self._subscriber is not None: self.pub.unsubscribe(self._subscriber)
        self._subscriber = sub
        self.pub.subscribe(sub)
        print("STT → SBERT → PF. Speak; Ctrl+C to stop.", flush=True)
        try: self.pub.start()
        except KeyboardInterrupt: self.pub.stop()


def run_embedding_to_pf_subscriber(stt_config: Optional[STTConfig] = None, model_name: str = DEFAULT_MODEL_NAME,
    sbert_cache_dir: Optional[Union[Path, str]] = None,
    params_callback: Optional[Callable[[str, np.ndarray, Optional[dict], Optional[dict]], None]] = None,
    print_params: bool = True, use_ridge: bool = True, use_nn: bool = True) -> None:
    model = _load_sbert_model(model_name, sbert_cache_dir)
    pub = STTPublisher(stt_config or STTConfig())
    pub.subscribe(_make_on_transcript(model, print_params, params_callback, use_ridge, use_nn))
    print("STT → SBERT → PF. Speak; Ctrl+C to stop.", flush=True)
    try: pub.start()
    except KeyboardInterrupt: pub.stop()


if __name__ == "__main__":
    run_embedding_to_pf_subscriber()
