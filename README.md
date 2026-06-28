# SonicTrace

Audio analysis pipeline that performs speaker diarization, emotion detection, and transcription on a WAV, MP3, or FLAC file. Results are exported as JSON, SRT subtitles, CSV, and a plain-text transcript, with an optional visual timeline.

## Pipeline

```
Audio file
  └─ [1] Voice Activity Detection   (Silero VAD)
  └─ [2] Speaker Embeddings         (ECAPA-TDNN via SpeechBrain)
  └─ [3] Speaker Clustering         (Agglomerative + silhouette auto-detection)
  └─ [4] Emotion Detection          (Wav2Vec2 — 8 RAVDESS classes)
  └─ [5] Transcription              (OpenAI Whisper)
  └─ Merge → JSON + SRT + CSV + transcript + timeline plot
```

Detected emotions: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

## Outputs

| File | Description |
|------|-------------|
| `pipeline_results.json` | Full segment data (speaker, emotion, confidence, text, timestamps) |
| `timeline_plot.png` | Per-speaker Gantt chart with emotion labels |
| `output.srt` | Subtitle file with speaker labels |
| `output.csv` | Spreadsheet-friendly export |
| `transcript.txt` | Human-readable transcript grouped by speaker |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Models are downloaded automatically on first run.

## Usage

### CLI

```bash
# Run the full pipeline
python main.py audio.wav

# Choose a larger Whisper model for better accuracy
python main.py audio.wav --whisper-model small

# Available Whisper sizes: tiny | base | small | medium | large
```

### Web app

```bash
python app.py
# Open http://localhost:5000
```

Upload a WAV, MP3, or FLAC file (up to 200 MB). The web interface runs the same pipeline and lets you download results in SRT, CSV, or transcript format.

### Merge pre-computed results

```bash
# Combine diarization and emotion JSONs without re-running the full pipeline
python pipeline.py --merge diarization_results.json emotion_results.json

# Include transcription
python pipeline.py --merge diarization_results.json emotion_results.json \
  --transcription transcription_results.json
```

### Run individual stages

```bash
python models/vad_model.py          audio.wav
python models/embedding_model.py    audio.wav
python models/clustering_model.py
python models/emotion_model.py      audio.wav segments.json
python models/transcription_model.py audio.wav segments.json [tiny|base|small|medium|large]
```

## Models

| Stage | Model |
|-------|-------|
| VAD | `snakers4/silero-vad` |
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` (192-dim ECAPA-TDNN) |
| Emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| Transcription | OpenAI Whisper (`base` by default) |

## Dependencies

```
torch, torchaudio, speechbrain, transformers, openai-whisper,
scikit-learn, librosa, soundfile, numpy, matplotlib
```
