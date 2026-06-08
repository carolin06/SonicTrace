
import json
import os
import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample_poly
from math import gcd


SAMPLE_RATE = 16000
MODEL_SIZE = "base"   # tiny | base | small | medium | large
RESULTS_PATH = "transcription_results.json"


def _load_audio(audio_path: str) -> np.ndarray:
    data, sr = sf.read(audio_path, always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g)
    return data.astype(np.float32)


def load_model(size: str = MODEL_SIZE) -> whisper.Whisper:
    print(f"Loading Whisper ({size})...")
    return whisper.load_model(size)


def transcribe_segments(
    audio_path: str,
    segments: list[dict],
    model_size: str = MODEL_SIZE,
    language: str | None = None,
    save_path: str | None = RESULTS_PATH,
) -> list[dict]:
    """
    Transcribe each VAD segment individually using Whisper.

    Args:
        audio_path:  Path to a 16 kHz mono WAV file.
        segments:    List of {"start": float, "end": float} dicts (seconds).
        model_size:  Whisper model size — trade speed vs accuracy.
        language:    ISO-639-1 code (e.g. "en"). None lets Whisper auto-detect.
        save_path:   Where to write transcription_results.json; None to skip.

    Returns:
        List of {"start", "end", "text"} dicts in the same order as segments.
    """
    model = load_model(model_size)
    audio = _load_audio(audio_path)
    total_samples = len(audio)

    results = []
    for seg in segments:
        start = seg["start"]
        end = min(seg["end"], total_samples / SAMPLE_RATE)
        if end - start < 0.1:
            continue

        chunk = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]

        result = model.transcribe(
            chunk,
            fp16=False,
            language=language,
            verbose=False,
        )
        text = result["text"].strip()

        results.append({"start": start, "end": end, "text": text})

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {save_path}  — {len(results)} segment(s)")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcription_model.py <audio.wav> [segments.json] [model_size]")
        sys.exit(1)

    audio_path = sys.argv[1]
    seg_path   = sys.argv[2] if len(sys.argv) > 2 else "segments.json"
    size       = sys.argv[3] if len(sys.argv) > 3 else MODEL_SIZE

    with open(seg_path) as f:
        segments = json.load(f)

    print(f"Transcribing {len(segments)} segment(s) with Whisper-{size}...")
    results = transcribe_segments(audio_path, segments, model_size=size)

    print("\nTranscription results:")
    for r in results:
        print(f"  {r['start']:7.3f}s -> {r['end']:7.3f}s  |  {r['text']}")
