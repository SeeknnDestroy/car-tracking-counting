from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from car_tracking_counting.counting import CountEvent, TrackObservation

EVENT_FIELDNAMES = [
    "frame_index",
    "video_time_s",
    "track_id",
    "class_id",
    "line_name",
    "direction",
]

OBSERVATION_FIELDNAMES = [
    "frame_index",
    "track_id",
    "class_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
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


def load_events_csv(events_path: str | Path) -> list[dict[str, str]]:
    with Path(events_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_observations_csv(
    observations_path: str | Path,
) -> dict[int, list[TrackObservation]]:
    observations_by_frame: dict[int, list[TrackObservation]] = {}
    with Path(observations_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_index = int(row["frame_index"])
            observations_by_frame.setdefault(frame_index, []).append(
                TrackObservation(
                    track_id=int(row["track_id"]),
                    class_id=int(row["class_id"]),
                    bbox_xyxy=(
                        int(row["x1"]),
                        int(row["y1"]),
                        int(row["x2"]),
                        int(row["y2"]),
                    ),
                    centroid=(float(row["center_x"]), float(row["center_y"])),
                )
            )
    return observations_by_frame
