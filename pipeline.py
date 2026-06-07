"""
SonicTrace pipeline — two modes:

  Full run (requires audio file):
      python pipeline.py audio.wav

  Merge only (pre-computed JSONs):
      python pipeline.py --merge diarization_results.json emotion_results.json
      python pipeline.py --merge diarization_results.json emotion_results.json --transcription transcription_results.json
"""

import argparse
import json
import os


DIARIZATION_PATH    = "diarization_results.json"
EMOTION_PATH        = "emotion_results.json"
TRANSCRIPTION_PATH  = "transcription_results.json"
PIPELINE_RESULTS_PATH = "pipeline_results.json"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_results(
    diarization: list[dict],
    emotions: list[dict],
    transcriptions: list[dict] | None = None,
) -> list[dict]:
    """
    Combine diarization, emotion, and (optionally) transcription into one
    record per segment. All lists must share the same segment order.
    Timestamps are validated to match within 1 ms tolerance.
    """
    if len(diarization) != len(emotions):
        raise ValueError(
            f"Length mismatch: {len(diarization)} diarization vs {len(emotions)} emotion records"
        )
    if transcriptions is not None and len(transcriptions) != len(diarization):
        raise ValueError(
            f"Length mismatch: {len(diarization)} diarization vs {len(transcriptions)} transcription records"
        )

    merged = []
    for i, (d, e) in enumerate(zip(diarization, emotions)):
        if abs(d["start"] - e["start"]) > 0.001 or abs(d["end"] - e["end"]) > 0.001:
            raise ValueError(
                f"Segment {i} timestamp mismatch — "
                f"diarization ({d['start']}-{d['end']}) vs emotion ({e['start']}-{e['end']})"
            )

        record = {
            "start":      d["start"],
            "end":        d["end"],
            "duration":   round(d["end"] - d["start"], 3),
            "speaker":    d["speaker"],
            "emotion":    e["emotion"],
            "confidence": e["confidence"],
        }

        if transcriptions is not None:
            t = transcriptions[i]
            if abs(d["start"] - t["start"]) > 0.001 or abs(d["end"] - t["end"]) > 0.001:
                raise ValueError(
                    f"Segment {i} timestamp mismatch — "
                    f"diarization ({d['start']}-{d['end']}) vs transcription ({t['start']}-{t['end']})"
                )
            record["text"] = t["text"]

        merged.append(record)

    return merged


def merge_from_files(
    diarization_path: str = DIARIZATION_PATH,
    emotion_path: str = EMOTION_PATH,
    transcription_path: str | None = None,
    save_path: str = PIPELINE_RESULTS_PATH,
) -> list[dict]:
    with open(diarization_path) as f:
        diarization = json.load(f)
    with open(emotion_path) as f:
        emotions = json.load(f)
    transcriptions = None
    if transcription_path:
        with open(transcription_path) as f:
            transcriptions = json.load(f)

    results = merge_results(diarization, emotions, transcriptions)
    _save(results, save_path)
    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    audio_path: str,
    whisper_model: str = "base",
    save_path: str = PIPELINE_RESULTS_PATH,
) -> list[dict]:
    """
    End-to-end: VAD → embeddings → clustering → emotion → transcription → merge.
    """
    from models.vad_model import detect_speech_segments
    from models.embedding_model import extract_embeddings
    from models.clustering_model import run_diarization
    from models.emotion_model import predict_emotions
    from models.transcription_model import transcribe_segments

    print("=" * 50)
    print("Step 1/5  VAD")
    print("=" * 50)
    segments = detect_speech_segments(audio_path)
    print(f"  {len(segments)} speech segment(s) detected")

    print("\n" + "=" * 50)
    print("Step 2/5  Speaker embeddings")
    print("=" * 50)
    embeddings, valid_segments = extract_embeddings(audio_path, segments)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\n" + "=" * 50)
    print("Step 3/5  Speaker clustering")
    print("=" * 50)
    diarization = run_diarization()

    print("\n" + "=" * 50)
    print("Step 4/5  Emotion detection")
    print("=" * 50)
    emotions = predict_emotions(audio_path, valid_segments)

    print("\n" + "=" * 50)
    print("Step 5/5  Transcription")
    print("=" * 50)
    transcriptions = transcribe_segments(audio_path, valid_segments, model_size=whisper_model)

    print("\n" + "=" * 50)
    print("Merging results")
    print("=" * 50)
    results = merge_results(diarization, emotions, transcriptions)
    _save(results, save_path)

    return results


# ---------------------------------------------------------------------------
# Save + display
# ---------------------------------------------------------------------------

def _save(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}  — {len(results)} segment(s)")


def print_summary(results: list[dict]) -> None:
    has_text = "text" in (results[0] if results else {})

    print("\nFinal results:")
    header = f"  {'Start':>8}  {'End':>8}  {'Dur':>6}  {'Speaker':<12}  {'Emotion':<12}  {'Conf':<6}"
    if has_text:
        header += "  Text"
    print(header)
    print("  " + "-" * (66 + (40 if has_text else 0)))

    for r in results:
        line = (
            f"  {r['start']:8.3f}  {r['end']:8.3f}  {r['duration']:6.2f}s"
            f"  {r['speaker']:<12}  {r['emotion']:<12}  {r['confidence']:.4f}"
        )
        if has_text:
            snippet = r["text"][:60] + ("…" if len(r["text"]) > 60 else "")
            line += f"  {snippet}"
        print(line)

    speakers = sorted({r["speaker"] for r in results})
    print("\nPer-speaker emotion breakdown:")
    from collections import Counter
    for spk in speakers:
        segs = [r for r in results if r["speaker"] == spk]
        counts = Counter(r["emotion"] for r in segs)
        breakdown = "  ".join(f"{e}×{n}" for e, n in counts.most_common())
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
        help="Merge pre-computed diarization and emotion JSON files",
    )
    parser.add_argument(
        "--transcription",
        metavar="TRANSCRIPTION",
        help="Optional transcription JSON to include in merge",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        metavar="SIZE",
        help="Whisper model size for full pipeline (default: base)",
    )
    parser.add_argument("--output", default=PIPELINE_RESULTS_PATH, help="Output JSON path")
    args = parser.parse_args()

    if args.merge:
        results = merge_from_files(
            args.merge[0],
            args.merge[1],
            transcription_path=args.transcription,
            save_path=args.output,
        )
    elif args.audio:
        results = run_full_pipeline(args.audio, args.whisper_model, args.output)
    else:
        parser.print_help()
        return

    print_summary(results)


if __name__ == "__main__":
    main()
