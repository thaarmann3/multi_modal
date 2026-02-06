import numpy as np
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer

# ----------------- MATLAB transfer functions -----------------
def tansig(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

def logsig(x):
    return 1.0 / (1.0 + np.exp(-x))

def purelin(x):
    return x

def poslin(x):
    return np.maximum(0.0, x)

ACT = {
    "tansig": tansig,
    "logsig": logsig,
    "purelin": purelin,
    "poslin": poslin,
}

# ----------------- utilities -----------------
def ensure_1d(a):
    return np.asarray(a).reshape(-1)

def _to_str_list(x):
    """Convert MATLAB cellstr/char arrays loaded by SciPy into a list[str]."""
    if isinstance(x, str):
        return [x]
    x = np.atleast_1d(x)

    out = []
    for s in x:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, bytes):
            out.append(s.decode("utf-8"))
        elif isinstance(s, np.ndarray):
            # could be char array or object
            if s.dtype.kind in ("U", "S"):
                out.append("".join(s.tolist()).strip())
            else:
                out.append("".join(chr(int(c)) for c in s.flatten()).strip())
        else:
            out.append(str(s))
    return out

def _get_scalar_str(v):
    """Get a single string from MATLAB char / cellstr."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        lst = _to_str_list(v)
        return lst[0] if len(lst) else ""
    return str(v)

def _reshape_W3(W3, H2):
    """
    Force W3 into shape (1, H2, 4).
    Possible SciPy-loaded shapes:
      - (1, H2, 4)  (ideal)
      - (H2, 4)     (squeezed leading dim)
      - (4, H2)     (transposed)
      - (H2*4,) or (H2*4,1) flattened (rare)
    """
    W3 = np.asarray(W3, dtype=np.float64)

    if W3.ndim == 3:
        # Could still be (H2,1,4) etc. Try to coerce.
        if W3.shape[0] == 1 and W3.shape[1] == H2 and W3.shape[2] == 4:
            return W3
        if W3.shape[0] == H2 and W3.shape[1] == 1 and W3.shape[2] == 4:
            return np.transpose(W3, (1, 0, 2))
        if W3.shape[0] == 4 and W3.shape[1] == H2 and W3.shape[2] == 1:
            return np.transpose(W3, (2, 1, 0))
        # Fall through to generic reshape if weird
        W3 = W3.reshape(-1)

    if W3.ndim == 2:
        # (H2, 4) -> (1,H2,4)
        if W3.shape == (H2, 4):
            return W3[np.newaxis, :, :]
        # (4, H2) -> transpose then add axis
        if W3.shape == (4, H2):
            return W3.T[np.newaxis, :, :]
        # (1, H2*4) or (H2*4,1)
        if 1 in W3.shape:
            return W3.reshape(-1)

    # 1D fallback: must be H2*4
    flat = W3.reshape(-1)
    if flat.size != H2 * 4:
        raise ValueError(f"Cannot reshape W3: got size {flat.size}, expected {H2*4}")
    return flat.reshape(1, H2, 4, order="F")  # MATLAB column-major

def _reshape_W_general(W, expected_first, expected_second):
    """
    Force W into shape (expected_first, expected_second, 4) using MATLAB-like ordering.
    Handles cases where SciPy squeezes or flattens.
    """
    W = np.asarray(W, dtype=np.float64)

    if W.ndim == 3:
        # If last dim is 4, great; otherwise try permutes
        if W.shape[0] == expected_first and W.shape[1] == expected_second and W.shape[2] == 4:
            return W
        # Sometimes loaded as (4, expected_first, expected_second)
        if W.shape[0] == 4 and W.shape[1] == expected_first and W.shape[2] == expected_second:
            return np.transpose(W, (1, 2, 0))
        # Sometimes loaded as (expected_second, expected_first, 4)
        if W.shape[0] == expected_second and W.shape[1] == expected_first and W.shape[2] == 4:
            return np.transpose(W, (1, 0, 2))
        # Fallback flatten
        W = W.reshape(-1)

    if W.ndim == 2:
        # If (expected_first, expected_second*4) etc, flatten
        if W.shape == (expected_first, expected_second):
            # Not possible here because we need 4 outputs; treat as single-output and error
            raise ValueError(f"W looks like single-output weights: {W.shape}")
        W = W.reshape(-1)

    flat = W.reshape(-1)
    if flat.size != expected_first * expected_second * 4:
        raise ValueError(
            f"Cannot reshape W: got size {flat.size}, expected {expected_first*expected_second*4}"
        )
    # MATLAB stores column-major; preserve that with order='F'
    return flat.reshape(expected_first, expected_second, 4, order="F")

def _reshape_b(b, H):
    """
    Force b into shape (H, 4).
    Possible shapes: (H,4), (4,H), (H*4,), etc.
    """
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 2:
        if b.shape == (H, 4):
            return b
        if b.shape == (4, H):
            return b.T
        b = b.reshape(-1)

    flat = b.reshape(-1)
    if flat.size != H * 4:
        raise ValueError(f"Cannot reshape b: got size {flat.size}, expected {H*4}")
    return flat.reshape(H, 4, order="F")

# ----------------- load exported MATLAB model -----------------
mat = loadmat("exported_models_like_cv.mat", squeeze_me=True, struct_as_record=False)

muX  = ensure_1d(mat["muX"]).astype(np.float64)
sigX = ensure_1d(mat["sigX"]).astype(np.float64)
muY  = ensure_1d(mat["muY"]).astype(np.float64)
sigY = ensure_1d(mat["sigY"]).astype(np.float64)

B_ridge = np.asarray(mat["B_ridge"], dtype=np.float64)  # (p+1, 4)

yNames = _to_str_list(mat.get("yNames_cell", ["x_m", "y_m", "amplitude", "radius"]))
tf1    = _to_str_list(mat["tf1_cell"])
tf2    = _to_str_list(mat["tf2_cell"])
tf3    = _to_str_list(mat["tf3_cell"])

nnBackend = _get_scalar_str(mat.get("nnBackend_char", "feedforwardnet"))
if nnBackend != "feedforwardnet":
    raise RuntimeError(f"Expected feedforwardnet export, got nnBackend={nnBackend}")

W1_raw = mat["W1"]; b1_raw = mat["b1"]
W2_raw = mat["W2"]; b2_raw = mat["b2"]
W3_raw = mat["W3"]; b3_raw = mat["b3"]

# Infer dimensions
p = muX.shape[0]  # embedding dimension
# W1 should be (H1,p,4) but might load differently. We can infer H1 from b1.
b1_tmp = np.asarray(b1_raw, dtype=np.float64).reshape(-1)
# b1 has H1*4 entries
H1 = int(b1_tmp.size // 4)
b2_tmp = np.asarray(b2_raw, dtype=np.float64).reshape(-1)
H2 = int(b2_tmp.size // 4)

# Reshape weights/biases robustly
W1 = _reshape_W_general(W1_raw, H1, p)
b1 = _reshape_b(b1_raw, H1)

W2 = _reshape_W_general(W2_raw, H2, H1)
b2 = _reshape_b(b2_raw, H2)

W3 = _reshape_W3(W3_raw, H2)          # (1, H2, 4)
b3 = np.asarray(b3_raw, dtype=np.float64).reshape(-1)
if b3.size == 4:
    b3 = b3.reshape(1, 4)
elif b3.size == 1 * 4:
    b3 = b3.reshape(1, 4, order="F")
else:
    raise ValueError(f"b3 has unexpected size {b3.size}, expected 4")

# ----------------- SBERT embedder (must match embeddings CSV model) -----------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
embedder = SentenceTransformer(MODEL_NAME)

def embed_sentence(sentence: str) -> np.ndarray:
    emb = embedder.encode([sentence], convert_to_numpy=True, normalize_embeddings=False)[0]
    return emb.astype(np.float64)

def standardize_x(x: np.ndarray) -> np.ndarray:
    return (x - muX) / sigX

# ----------------- Ridge inference (predicts Yz -> unstandardize) -----------------
def ridge_predict(xz: np.ndarray) -> np.ndarray:
    x1 = np.concatenate([np.ones(1), xz])   # (p+1,)
    yz_hat = x1 @ B_ridge                   # (4,) in standardized Y units
    return muY + sigY * yz_hat

# ----------------- NN inference (predicts y_z per output -> unstandardize) -----------------
def nn_forward_one(xz: np.ndarray, j: int) -> float:
    f1 = ACT[tf1[j]]
    f2 = ACT[tf2[j]]
    f3 = ACT[tf3[j]]

    a1 = f1(W1[:, :, j] @ xz + b1[:, j])          # (H1,)
    a2 = f2(W2[:, :, j] @ a1 + b2[:, j])          # (H2,)
    yz = f3(W3[0, :, j] @ a2 + b3[0, j])          # scalar
    y = muY[j] + sigY[j] * yz
    return float(y)

def nn_predict(xz: np.ndarray) -> np.ndarray:
    return np.array([nn_forward_one(xz, j) for j in range(4)], dtype=np.float64)

def predict(sentence: str):
    x = embed_sentence(sentence)
    if x.shape[0] != p:
        raise ValueError(
            f"Embedding dim mismatch: got {x.shape[0]} but model expects {p}. "
            f"Check SBERT model name used for embeddings CSV."
        )
    xz = standardize_x(x)
    y_ridge = ridge_predict(xz)
    y_nn    = nn_predict(xz)
    return dict(zip(yNames, y_ridge)), dict(zip(yNames, y_nn))

if __name__ == "__main__":
    # Quick sanity prints
    print("Loaded:", len(yNames), "outputs:", yNames)
    print("p=", p, "H1=", H1, "H2=", H2)
    print("W1", W1.shape, "W2", W2.shape, "W3", W3.shape)

    s = "Let's bring the grip forward and to the right moderately now."
    ridge_out, nn_out = predict(s)
    print("\nSentence:", s)
    print("Ridge:", ridge_out)
    print("NN:", nn_out)
