import csv
import json
import sys
from datetime import timedelta


INPUT_FILE = "pipeline_results.json"
SRT_FILE = "output.srt"
CSV_FILE = "output.csv"
TRANSCRIPT_FILE = "transcript.txt"


def load_results(path: str = INPUT_FILE) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _srt_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_s = int(td.total_seconds())
    ms = int(round((td.total_seconds() - total_s) * 1000))
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_srt(results: list[dict], path: str = SRT_FILE) -> None:
    lines = []
    for i, r in enumerate(results, start=1):
        text = r.get("text", f"[{r['emotion']}]")
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(r['start'])} --> {_srt_timestamp(r['end'])}")
        lines.append(f"[{r['speaker']}] {text}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}  ({len(results)} subtitle entries)")


def export_csv(results: list[dict], path: str = CSV_FILE) -> None:
    fieldnames = ["speaker", "start", "end", "duration", "emotion", "confidence", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {path}  ({len(results)} rows)")


def export_transcript(results: list[dict], path: str = TRANSCRIPT_FILE) -> None:
    lines = []
    current_speaker = None
    for r in results:
        spk = r["speaker"]
        text = r.get("text", "").strip()
        emotion = r["emotion"]
        ts = f"[{r['start']:.1f}s]"

        if spk != current_speaker:
            if lines:
                lines.append("")
            lines.append(f"{spk}:")
            current_speaker = spk

        lines.append(f"  {ts} ({emotion})  {text}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}  ({len(results)} segments)")


def export_all(
    results: list[dict],
    srt_path: str = SRT_FILE,
    csv_path: str = CSV_FILE,
    transcript_path: str = TRANSCRIPT_FILE,
) -> None:
    export_srt(results, srt_path)
    export_csv(results, csv_path)
    export_transcript(results, transcript_path)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    results = load_results(input_path)
    print(f"Loaded {len(results)} segments from {input_path}\n")
    export_all(results)
