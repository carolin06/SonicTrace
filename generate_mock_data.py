import json
import os
import numpy as np

SEED = 42
N_SEGMENTS = 20
N_SPEAKERS = 3
EMBEDDING_DIM = 192
SAVE_DIR = "."

rng = np.random.default_rng(SEED)


def _make_cluster_centers(n: int, dim: int) -> np.ndarray:
    """Return n well-separated unit-vector cluster centers via Gram-Schmidt."""
    raw = rng.standard_normal((n, dim))
    centers = np.zeros_like(raw)
    for i in range(n):
        v = raw[i].copy()
        for j in range(i):
            v -= np.dot(v, centers[j]) * centers[j]
        centers[i] = v / np.linalg.norm(v)
    return centers


def generate_embeddings(
    n_segments: int = N_SEGMENTS,
    n_speakers: int = N_SPEAKERS,
    dim: int = EMBEDDING_DIM,
    noise_scale: float = 0.08,
) -> tuple[np.ndarray, list[int]]:
    """
    Build (n_segments, dim) embeddings with tight, well-separated clusters.

    noise_scale controls intra-speaker spread; 0.08 gives cosine similarity
    within ~0.95 inside a cluster and ~0.3-0.5 across clusters — realistic
    for ECAPA-TDNN outputs.
    """
    centers = _make_cluster_centers(n_speakers, dim)

    # distribute segments roughly evenly, then shuffle for realistic turn order
    base = n_segments // n_speakers
    remainder = n_segments % n_speakers
    counts = [base + (1 if i < remainder else 0) for i in range(n_speakers)]

    labels = []
    for speaker_id, count in enumerate(counts):
        labels.extend([speaker_id] * count)
    rng.shuffle(labels)

    embeddings = np.zeros((n_segments, dim), dtype=np.float32)
    for i, spk in enumerate(labels):
        noise = rng.standard_normal(dim) * noise_scale
        v = centers[spk] + noise
        embeddings[i] = (v / np.linalg.norm(v)).astype(np.float32)

    return embeddings, labels


def generate_segments(
    n_segments: int = N_SEGMENTS,
    min_duration: float = 1.0,
    max_duration: float = 5.0,
    min_gap: float = 0.2,
    max_gap: float = 1.5,
) -> list[dict]:
    """Return chronological {"start": float, "end": float} segment dicts."""
    segments = []
    cursor = 0.0
    for _ in range(n_segments):
        duration = rng.uniform(min_duration, max_duration)
        end = cursor + duration
        segments.append({"start": round(cursor, 3), "end": round(end, 3)})
        gap = rng.uniform(min_gap, max_gap)
        cursor = end + gap
    return segments


def main() -> None:
    embeddings, labels = generate_embeddings()
    segments = generate_segments()

    os.makedirs(SAVE_DIR, exist_ok=True)
    emb_path = os.path.join(SAVE_DIR, "embeddings.npy")
    seg_path = os.path.join(SAVE_DIR, "segments.json")

    np.save(emb_path, embeddings)
    with open(seg_path, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"Saved embeddings.npy  — shape: {embeddings.shape}")
    print(f"Saved segments.json   — {len(segments)} segment(s)")
    print(f"\nGround-truth speaker distribution:")
    for spk in range(N_SPEAKERS):
        count = labels.count(spk)
        indices = [i for i, l in enumerate(labels) if l == spk]
        print(f"  Speaker_{spk + 1}: {count} segments  (indices {indices})")

    # sanity check: intra vs inter cosine similarity
    print("\nCosine similarity sanity check:")
    from sklearn.preprocessing import normalize
    normed = normalize(embeddings)
    sim_matrix = normed @ normed.T
    for spk in range(N_SPEAKERS):
        idx = [i for i, l in enumerate(labels) if l == spk]
        if len(idx) > 1:
            pairs = [(i, j) for i in idx for j in idx if i < j]
            intra = np.mean([sim_matrix[i, j] for i, j in pairs])
            print(f"  Speaker_{spk + 1} intra-cluster cosine sim: {intra:.3f}")
    all_inter = []
    for a in range(N_SPEAKERS):
        for b in range(a + 1, N_SPEAKERS):
            idx_a = [i for i, l in enumerate(labels) if l == a]
            idx_b = [i for i, l in enumerate(labels) if l == b]
            pairs = [(i, j) for i in idx_a for j in idx_b]
            all_inter.extend([sim_matrix[i, j] for i, j in pairs])
    print(f"  Inter-cluster cosine sim (avg): {np.mean(all_inter):.3f}")


if __name__ == "__main__":
    main()
