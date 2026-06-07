import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


INPUT_FILE = "pipeline_results.json"
OUTPUT_FILE = "timeline_plot.png"

SPEAKER_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def load_results(path: str = INPUT_FILE) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_timeline(results: list[dict], output_path: str = OUTPUT_FILE) -> None:
    speakers = sorted({r["speaker"] for r in results})
    spk_index = {spk: i for i, spk in enumerate(speakers)}
    color_map = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, spk in enumerate(speakers)}

    total_duration = max(r["end"] for r in results)
    n_speakers = len(speakers)

    fig_width = max(14, total_duration * 0.4)
    fig_height = max(4, n_speakers * 1.4 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_height = 0.5

    for r in results:
        y = spk_index[r["speaker"]]
        start = r["start"]
        dur = r["duration"]
        color = color_map[r["speaker"]]
        emotion = r.get("emotion", "")

        ax.barh(
            y, dur, left=start, height=bar_height,
            color=color, edgecolor="white", linewidth=0.8, alpha=0.85,
        )

        # emotion label — only if block is wide enough to fit text
        if dur > 0.8:
            ax.text(
                start + dur / 2, y,
                emotion,
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                clip_on=True,
            )

    ax.set_yticks(range(n_speakers))
    ax.set_yticklabels(speakers, fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("SonicTrace — Speaker Timeline", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, total_duration * 1.02)
    ax.set_ylim(-0.6, n_speakers - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=color_map[spk], label=spk) for spk in speakers
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    results = load_results(input_path)
    print(f"Loaded {len(results)} segments from {input_path}")
    plot_timeline(results, output_path)
