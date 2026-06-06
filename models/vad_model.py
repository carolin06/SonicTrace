import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION = 0.5  # seconds


def load_audio(audio_path: str) -> torch.Tensor:
    data, sr = sf.read(audio_path, always_2d=True)  # (samples, channels)
    if data.shape[1] > 1:
        data = data.mean(axis=1)  # mix to mono
    else:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return torch.from_numpy(data.astype(np.float32))  # shape: (num_samples,)


def detect_speech_segments(audio_path: str) -> list[dict]:
    """
    Run Silero VAD on a WAV file and return speech segments.

    Returns a list of {"start": float, "end": float} dicts in seconds,
    excluding segments shorter than MIN_SEGMENT_DURATION.
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps, _, _, _, _ = utils

    waveform = load_audio(audio_path)

    raw_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
    )

    segments = []
    for ts in raw_timestamps:
        start, end = ts["start"], ts["end"]
        if (end - start) >= MIN_SEGMENT_DURATION:
            segments.append({"start": start, "end": end})

    return segments


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python vad_model.py <audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    segments = detect_speech_segments(path)
    print(f"Detected {len(segments)} speech segment(s):")
    print(json.dumps(segments, indent=2))
