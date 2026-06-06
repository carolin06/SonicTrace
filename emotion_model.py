import json
import os
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification


SAMPLE_RATE = 16000
MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# RAVDESS labels produced by this checkpoint:
# angry, calm, disgust, fearful, happy, neutral, sad, surprised
RESULTS_PATH = "emotion_results.json"


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


def load_model(
    model_id: str = MODEL_ID,
) -> tuple[AutoFeatureExtractor, Wav2Vec2ForSequenceClassification]:
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return extractor, model


def _predict_single(
    chunk: np.ndarray,
    extractor: AutoFeatureExtractor,
    model: Wav2Vec2ForSequenceClassification,
) -> tuple[str, float]:
    """Return (emotion_label, confidence) for one audio chunk."""
    inputs = extractor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_labels)

    probs = torch.softmax(logits, dim=-1).squeeze()
    top_idx = probs.argmax().item()
    label = model.config.id2label[top_idx]
    confidence = round(probs[top_idx].item(), 4)
    return label, confidence


def predict_emotions(
    audio_path: str,
    segments: list[dict],
    model_id: str = MODEL_ID,
    save_path: str | None = RESULTS_PATH,
) -> list[dict]:
    """
    Predict emotion for each segment in the audio file.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.
        segments:   List of {"start": float, "end": float} dicts (seconds).
        model_id:   HuggingFace model ID for Wav2Vec2 emotion classifier.
        save_path:  Where to write emotion_results.json; pass None to skip.

    Returns:
        List of dicts: {"start", "end", "emotion", "confidence"}.
    """
    extractor, model = load_model(model_id)
    audio = _load_audio(audio_path)
    total_samples = len(audio)

    results = []
    for seg in segments:
        start, end = seg["start"], min(seg["end"], total_samples / SAMPLE_RATE)
        if end - start < 0.1:
            continue

        s, e = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        chunk = audio[s:e]

        emotion, confidence = _predict_single(chunk, extractor, model)
        results.append({
            "start": start,
            "end": end,
            "emotion": emotion,
            "confidence": confidence,
        })

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {save_path}  — {len(results)} segment(s)")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python emotion_model.py <audio.wav> [segments.json]")
        sys.exit(1)

    audio_path = sys.argv[1]
    seg_path = sys.argv[2] if len(sys.argv) > 2 else "segments.json"

    with open(seg_path) as f:
        segments = json.load(f)

    print(f"Predicting emotions for {len(segments)} segment(s)...")
    results = predict_emotions(audio_path, segments)

    print("\nEmotion predictions:")
    for r in results:
        print(
            f"  {r['start']:7.3f}s -> {r['end']:7.3f}s"
            f"  |  {r['emotion']:<12}  conf: {r['confidence']:.4f}"
        )
