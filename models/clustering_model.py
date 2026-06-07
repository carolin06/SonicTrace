import json
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


EMBEDDINGS_PATH = "embeddings.npy"
SEGMENTS_PATH = "segments.json"
RESULTS_PATH = "diarization_results.json"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6


def load_inputs(
    embeddings_path: str = EMBEDDINGS_PATH,
    segments_path: str = SEGMENTS_PATH,
) -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(embeddings_path)
    with open(segments_path) as f:
        segments = json.load(f)
    assert embeddings.shape[0] == len(segments), (
        f"Mismatch: {embeddings.shape[0]} embeddings vs {len(segments)} segments"
    )
    return embeddings, segments


def detect_num_speakers(
    embeddings: np.ndarray,
    min_k: int = MIN_SPEAKERS,
    max_k: int = MAX_SPEAKERS,
) -> int:
    """
    Try k = min_k..max_k with agglomerative clustering (cosine distance) and
    return the k with the highest silhouette score.
    """
    max_k = min(max_k, len(embeddings) - 1)
    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        labels = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        ).fit_predict(embeddings)

        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Auto-detected speakers: {best_k}  (silhouette score: {best_score:.4f})")
    return best_k


def cluster_speakers(
    embeddings: np.ndarray,
    n_speakers: int,
) -> np.ndarray:
    """
    Cluster normalized embeddings with agglomerative clustering (cosine distance).
    Returns integer label array of shape (N,).
    """
    normed = normalize(embeddings, norm="l2")
    labels = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    ).fit_predict(normed)
    return labels


def build_results(
    segments: list[dict],
    labels: np.ndarray,
) -> list[dict]:
    """Attach Speaker_N labels to each segment."""
    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "speaker": f"Speaker_{int(label) + 1}",
        }
        for seg, label in zip(segments, labels)
    ]


def save_results(results: list[dict], path: str = RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}  — {len(results)} segment(s)")


def run_diarization(
    embeddings_path: str = EMBEDDINGS_PATH,
    segments_path: str = SEGMENTS_PATH,
    results_path: str = RESULTS_PATH,
) -> list[dict]:
    """
    Full pipeline: load → normalize → auto-detect speakers → cluster → save.
    Returns the annotated results list.
    """
    embeddings, segments = load_inputs(embeddings_path, segments_path)
    print(f"Loaded embeddings: {embeddings.shape}  |  segments: {len(segments)}")

    if len(segments) < 2:
        print("Only 1 segment — skipping clustering, assigning Speaker_1")
        results = [{"start": segments[0]["start"], "end": segments[0]["end"], "speaker": "Speaker_1"}]
        save_results(results, results_path)
        print("\nDiarization results:")
        for r in results:
            print(f"  {r['start']:7.3f}s -> {r['end']:7.3f}s  |  {r['speaker']}")
        return results

    n_speakers = detect_num_speakers(embeddings)
    labels = cluster_speakers(embeddings, n_speakers)
    results = build_results(segments, labels)
    save_results(results, results_path)

    print("\nDiarization results:")
    for r in results:
        print(f"  {r['start']:7.3f}s -> {r['end']:7.3f}s  |  {r['speaker']}")

    return results


if __name__ == "__main__":
    run_diarization()
