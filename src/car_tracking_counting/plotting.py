from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _load_events(events_path: Path) -> list[dict[str, str]]:
    with events_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def generate_plots(events_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    events_file = Path(events_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    events = _load_events(events_file)

    over_time_path = output_root / "counts_over_time.png"
    by_direction_path = output_root / "counts_by_direction.png"

    if not events:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Counts Over Time")
        ax.set_xlabel("Video time (s)")
        ax.set_ylabel("Cumulative count")
        ax.text(0.5, 0.5, "No events recorded", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(over_time_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Counts by Direction")
        ax.set_xlabel("Direction")
        ax.set_ylabel("Count")
        ax.text(0.5, 0.5, "No events recorded", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(by_direction_path)
        plt.close(fig)
        return {
            "counts_over_time": over_time_path,
            "counts_by_direction": by_direction_path,
        }

    grouped_times: dict[str, list[float]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for event in events:
        series_name = f"{event['line_name']}:{event['direction']}"
        grouped_times[series_name].append(float(event["video_time_s"]))
        totals[event["direction"]] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    for series_name, times in sorted(grouped_times.items()):
        ordered_times = sorted(times)
        ax.step(ordered_times, range(1, len(ordered_times) + 1), where="post", label=series_name)
    ax.set_title("Counts Over Time")
    ax.set_xlabel("Video time (s)")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(over_time_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    directions = sorted(totals)
    ax.bar(directions, [totals[direction] for direction in directions])
    ax.set_title("Counts by Direction")
    ax.set_xlabel("Direction")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(by_direction_path)
    plt.close(fig)

    return {
        "counts_over_time": over_time_path,
        "counts_by_direction": by_direction_path,
    }
