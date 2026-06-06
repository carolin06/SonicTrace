"""
SonicTrace pipeline — two modes:

  Full run (requires audio file):
      python pipeline.py audio.wav

  Merge only (pre-computed JSONs):
      python pipeline.py --merge diarization_results.json emotion_results.json
"""

import argparse
import json
import os
import numpy as np


DIARIZATION_PATH = "diarization_results.json"
EMOTION_PATH = "emotion_results.json"
PIPELINE_RESULTS_PATH = "pipeline_results.json"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_results(
    diarization: list[dict],
    emotions: list[dict],
) -> list[dict]:
    """
    Combine diarization and emotion lists into one record per segment.

    Both lists must be ordered identically (same underlying segments.json).
    Timestamps are validated to match within 1 ms tolerance.
    """
    if len(diarization) != len(emotions):
        raise ValueError(
            f"Length mismatch: {len(diarization)} diarization vs {len(emotions)} emotion records"
        )

    merged = []
    for i, (d, e) in enumerate(zip(diarization, emotions)):
        if abs(d["start"] - e["start"]) > 0.001 or abs(d["end"] - e["end"]) > 0.001:
            raise ValueError(
                f"Segment {i} timestamp mismatch — "
                f"diarization ({d['start']}-{d['end']}) vs emotion ({e['start']}-{e['end']})"
            )
        merged.append({
            "start":      d["start"],
            "end":        d["end"],
            "duration":   round(d["end"] - d["start"], 3),
            "speaker":    d["speaker"],
            "emotion":    e["emotion"],
            "confidence": e["confidence"],
        })

    return merged


def merge_from_files(
    diarization_path: str = DIARIZATION_PATH,
    emotion_path: str = EMOTION_PATH,
    save_path: str = PIPELINE_RESULTS_PATH,
) -> list[dict]:
    with open(diarization_path) as f:
        diarization = json.load(f)
    with open(emotion_path) as f:
        emotions = json.load(f)

    results = merge_results(diarization, emotions)
    _save(results, save_path)
    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    audio_path: str,
    save_path: str = PIPELINE_RESULTS_PATH,
) -> list[dict]:
    """
    End-to-end: VAD → embeddings → clustering → emotion → merge.
    Intermediate files (embeddings.npy, segments.json, diarization/emotion
    JSONs) are written alongside the final pipeline_results.json.
    """
    from models.vad_model import detect_speech_segments
    from models.embedding_model import extract_embeddings
    from models.clustering_model import run_diarization
    from models.emotion_model import predict_emotions

    print("=" * 50)
    print("Step 1/4  VAD")
    print("=" * 50)
    segments = detect_speech_segments(audio_path)
    print(f"  {len(segments)} speech segment(s) detected")

    print("\n" + "=" * 50)
    print("Step 2/4  Speaker embeddings")
    print("=" * 50)
    embeddings, valid_segments = extract_embeddings(audio_path, segments)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\n" + "=" * 50)
    print("Step 3/4  Speaker clustering")
    print("=" * 50)
    diarization = run_diarization()

    print("\n" + "=" * 50)
    print("Step 4/4  Emotion detection")
    print("=" * 50)
    emotions = predict_emotions(audio_path, valid_segments)

    print("\n" + "=" * 50)
    print("Merging results")
    print("=" * 50)
    results = merge_results(diarization, emotions)
    _save(results, save_path)

    return results


# ---------------------------------------------------------------------------
# Save + display
# ---------------------------------------------------------------------------

def _save(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}  — {len(results)} segment(s)")


def print_summary(results: list[dict]) -> None:
    print("\nFinal results:")
    print(f"  {'Start':>8}  {'End':>8}  {'Dur':>6}  {'Speaker':<12}  {'Emotion':<12}  Conf")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r['start']:8.3f}  {r['end']:8.3f}  {r['duration']:6.2f}s"
            f"  {r['speaker']:<12}  {r['emotion']:<12}  {r['confidence']:.4f}"
        )

    # per-speaker emotion breakdown
    speakers = sorted({r["speaker"] for r in results})
    print("\nPer-speaker emotion breakdown:")
    for spk in speakers:
        segs = [r for r in results if r["speaker"] == spk]
        from collections import Counter
        counts = Counter(r["emotion"] for r in segs)
        top = counts.most_common()
        breakdown = "  ".join(f"{e}×{n}" for e, n in top)
        print(f"  {spk}: {breakdown}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SonicTrace pipeline")
    parser.add_argument("audio", nargs="?", help="Path to WAV file (full pipeline)")
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("DIARIZATION", "EMOTION"),
        help="Merge two pre-computed JSON files instead of running the full pipeline",
    )
    parser.add_argument("--output", default=PIPELINE_RESULTS_PATH, help="Output JSON path")
    args = parser.parse_args()

    if args.merge:
        results = merge_from_files(args.merge[0], args.merge[1], args.output)
    elif args.audio:
        results = run_full_pipeline(args.audio, args.output)
    else:
        parser.print_help()
        return

    print_summary(results)


if __name__ == "__main__":
    main()
