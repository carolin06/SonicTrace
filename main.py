import argparse
import json
import sys
from collections import Counter


def run(audio_path: str, whisper_model: str = "base") -> None:
    # --- import order matters: transformers before speechbrain ---
    from models.emotion_model import predict_emotions
    from models.transcription_model import transcribe_segments
    from models.vad_model import detect_speech_segments
    from models.embedding_model import extract_embeddings
    from models.clustering_model import run_diarization
    from pipeline import merge_results, _save
    from visualize import plot_timeline
    from export import export_all

    print("=" * 54)
    print("  SonicTrace — full pipeline")
    print("=" * 54)

    # Step 1 — VAD
    print("\n[1/5] Voice activity detection")
    segments = detect_speech_segments(audio_path)
    print(f"      {len(segments)} speech segment(s) found")

    # Step 2 — Embeddings
    print("\n[2/5] Speaker embeddings (ECAPA-TDNN)")
    embeddings, valid_segments = extract_embeddings(audio_path, segments)
    print(f"      Embeddings: {embeddings.shape}")

    # Step 3 — Clustering / diarization
    print("\n[3/5] Speaker clustering")
    diarization = run_diarization()

    # Step 4 — Emotion
    print("\n[4/5] Emotion detection (Wav2Vec2)")
    emotions = predict_emotions(audio_path, valid_segments)

    # Step 5 — Transcription
    print(f"\n[5/5] Transcription (Whisper {whisper_model})")
    transcriptions = transcribe_segments(audio_path, valid_segments, model_size=whisper_model)

    # Merge
    print("\n[+] Merging results")
    results = merge_results(diarization, emotions, transcriptions)
    _save(results, "pipeline_results.json")

    # Visualize
    print("\n[+] Generating timeline plot")
    plot_timeline(results, "timeline_plot.png")

    # Export
    print("\n[+] Exporting formats")
    export_all(results)

    # Summary
    speakers = sorted({r["speaker"] for r in results})
    emotion_counts = Counter(r["emotion"] for r in results)
    top_emotions = ", ".join(f"{e}({n})" for e, n in emotion_counts.most_common())

    print("\n" + "=" * 54)
    print("  Done")
    print("=" * 54)
    print(f"  Audio file    : {audio_path}")
    print(f"  Segments      : {len(results)}")
    print(f"  Speakers      : {len(speakers)} — {', '.join(speakers)}")
    print(f"  Emotions found: {top_emotions}")
    print(f"  Files saved   :")
    for f in ["pipeline_results.json", "timeline_plot.png",
              "output.srt", "output.csv", "transcript.txt"]:
        print(f"    - {f}")
    print("=" * 54)


def main() -> None:
    parser = argparse.ArgumentParser(description="SonicTrace — speaker diarization pipeline")
    parser.add_argument("audio", help="Path to input WAV file")
    parser.add_argument(
        "--whisper-model",
        default="base",
        metavar="SIZE",
        help="Whisper model size: tiny/base/small/medium/large (default: base)",
    )
    args = parser.parse_args()
    run(args.audio, whisper_model=args.whisper_model)


if __name__ == "__main__":
    main()
