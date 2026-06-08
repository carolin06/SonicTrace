import json
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pre-import in correct order: transformers before speechbrain
# (prevents k2 lazy-import crash when torch.distributed.tensor initialises)
# ---------------------------------------------------------------------------
print("Loading models — this may take a moment on first run...")
from models.emotion_model import predict_emotions          # noqa: E402
from models.transcription_model import transcribe_segments # noqa: E402
from models.vad_model import detect_speech_segments        # noqa: E402
from models.embedding_model import extract_embeddings      # noqa: E402
from models.clustering_model import run_diarization        # noqa: E402
from pipeline import merge_results, _save                  # noqa: E402
from export import export_all                              # noqa: E402
print("Models ready.")


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _to_wav(src_path: str) -> str:
    """Convert any supported audio format to 16 kHz mono WAV. Returns wav path."""
    if src_path.endswith(".wav"):
        return src_path
    import numpy as np
    import soundfile as sf
    import librosa
    wav_path = src_path.rsplit(".", 1)[0] + ".wav"
    y, sr = librosa.load(src_path, sr=16000, mono=True)
    sf.write(wav_path, y.astype("float32"), 16000)
    return wav_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    f = request.files["audio"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type. Use .wav, .mp3, or .flac"}), 400

    # Save upload
    ext = os.path.splitext(f.filename.lower())[1]
    uid = uuid.uuid4().hex[:8]
    raw_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    f.save(raw_path)

    try:
        audio_path = _to_wav(raw_path)

        # Run pipeline
        segments = detect_speech_segments(audio_path)
        embeddings, valid_segments = extract_embeddings(audio_path, segments)
        diarization = run_diarization()
        emotions = predict_emotions(audio_path, valid_segments)
        transcriptions = transcribe_segments(audio_path, valid_segments, model_size="base")

        results = merge_results(diarization, emotions, transcriptions)
        _save(results, "pipeline_results.json")
        export_all(results)

        # Compute summary stats
        speakers = sorted({r["speaker"] for r in results})
        total_duration = round(max(r["end"] for r in results), 2) if results else 0

        return jsonify({
            "segments": results,
            "summary": {
                "speaker_count": len(speakers),
                "speakers": speakers,
                "segment_count": len(results),
                "total_duration": total_duration,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up upload
        for p in [raw_path]:
            if os.path.exists(p) and p != audio_path:
                os.remove(p)


@app.route("/download/<fmt>")
def download(fmt: str):
    files = {
        "srt":        ("output.srt",        "text/plain",       "subtitles.srt"),
        "csv":        ("output.csv",         "text/csv",         "results.csv"),
        "transcript": ("transcript.txt",     "text/plain",       "transcript.txt"),
    }
    if fmt not in files:
        return jsonify({"error": "Unknown format"}), 404
    path, mimetype, download_name = files[fmt]
    if not os.path.exists(path):
        return jsonify({"error": "File not ready — run analysis first"}), 404
    return send_file(path, mimetype=mimetype, as_attachment=True,
                     download_name=download_name)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
