import json
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


SAMPLE_RATE = 16000
EMBEDDING_DIM = 192
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


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


def _slice_segment(audio: np.ndarray, start: float, end: float) -> torch.Tensor:
    s = int(start * SAMPLE_RATE)
    e = int(end * SAMPLE_RATE)
    segment = audio[s:e]
    return torch.from_numpy(segment).unsqueeze(0)  # (1, time)


def extract_embeddings(
    audio_path: str,
    segments: list[dict],
    save_dir: str = ".",
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract ECAPA-TDNN speaker embeddings for each VAD segment.

    Args:
        audio_path: Path to 16kHz mono WAV file.
        segments:   List of {"start": float, "end": float} dicts (seconds).
        save_dir:   Directory to write embeddings.npy and segments.json.

    Returns:
        embeddings:      np.ndarray of shape (N, 192)
        valid_segments:  Segments that produced an embedding (same order).
    """
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir="pretrained_models/ecapa-tdnn",
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )

    audio = _load_audio(audio_path)
    total_samples = len(audio)

    embeddings = []
    valid_segments = []

    for seg in segments:
        start, end = seg["start"], seg["end"]

        # guard against segments that extend past audio end
        end = min(end, total_samples / SAMPLE_RATE)
        if end - start < 0.1:
            continue

        wav = _slice_segment(audio, start, end)
        with torch.no_grad():
            emb = classifier.encode_batch(wav)  # (1, 1, 192)

        emb_np = emb.squeeze().cpu().numpy()     # (192,)
        assert emb_np.shape == (EMBEDDING_DIM,), f"Unexpected embedding shape: {emb_np.shape}"

        embeddings.append(emb_np)
        valid_segments.append({"start": start, "end": end})

    if not embeddings:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []

    embeddings_array = np.stack(embeddings)  # (N, 192)

    # save outputs for Person 2
    import os
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings_array)
    with open(os.path.join(save_dir, "segments.json"), "w") as f:
        json.dump(valid_segments, f, indent=2)

    print(f"Saved embeddings.npy — shape: {embeddings_array.shape}")
    print(f"Saved segments.json  — {len(valid_segments)} segment(s)")

    return embeddings_array, valid_segments


if __name__ == "__main__":
    import sys
    from vad_model import detect_speech_segments

    if len(sys.argv) < 2:
        print("Usage: python embedding_model.py <audio.wav>")
        sys.exit(1)

    path = sys.argv[1]

    print("Running VAD...")
    segments = detect_speech_segments(path)
    print(f"VAD found {len(segments)} segment(s)")

    print("\nExtracting embeddings...")
    embeddings, valid_segments = extract_embeddings(path, segments)

    print(f"\nResults:")
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Valid segments   : {len(valid_segments)}")
    for i, seg in enumerate(valid_segments):
        print(f"  [{i}] {seg['start']:.2f}s -> {seg['end']:.2f}s  | emb norm: {np.linalg.norm(embeddings[i]):.4f}")
