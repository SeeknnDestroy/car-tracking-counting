from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from car_tracking_counting.counting import CountEvent

EVENT_FIELDNAMES = [
    "frame_index",
    "video_time_s",
    "track_id",
    "class_id",
    "line_name",
    "direction",
]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_events_csv(events: list[CountEvent], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_row())
    return output_path


def write_summary_json(summary: dict[str, Any], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
