import json
import sys
import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
SEGMENTS_FILE = "segments.json"
EMBEDDING_DIM = 192

results = []

def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append(passed)


# --- embeddings.npy ---
try:
    embeddings = np.load(EMBEDDINGS_FILE)
    check("embeddings.npy exists", True)
except FileNotFoundError:
    check("embeddings.npy exists", False, "file not found")
    embeddings = None
except Exception as e:
    check("embeddings.npy exists", False, str(e))
    embeddings = None

if embeddings is not None:
    shape_ok = embeddings.ndim == 2 and embeddings.shape[1] == EMBEDDING_DIM
    check(
        f"embeddings shape is (N, {EMBEDDING_DIM})",
        shape_ok,
        f"got {embeddings.shape}",
    )
    dtype_ok = embeddings.dtype == np.float32
    check("embeddings dtype is float32", dtype_ok, f"got {embeddings.dtype}")
    N = embeddings.shape[0]
else:
    N = None

# --- segments.json ---
try:
    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)
    check("segments.json exists", True)
except FileNotFoundError:
    check("segments.json exists", False, "file not found")
    segments = None
except Exception as e:
    check("segments.json exists", False, str(e))
    segments = None

if segments is not None and N is not None:
    count_ok = len(segments) == N
    check(
        "segment count matches N",
        count_ok,
        f"segments={len(segments)}, embeddings={N}",
    )

if segments is not None:
    all_keys_ok = all("start" in s and "end" in s for s in segments)
    check("every segment has start and end keys", all_keys_ok)

    all_order_ok = all(s["end"] > s["start"] for s in segments)
    bad = [i for i, s in enumerate(segments) if s["end"] <= s["start"]]
    check(
        "end > start for every segment",
        all_order_ok,
        f"bad indices: {bad}" if bad else "",
    )

# --- summary ---
print()
total = len(results)
passed = sum(results)
print(f"{passed}/{total} checks passed")
sys.exit(0 if all(results) else 1)
