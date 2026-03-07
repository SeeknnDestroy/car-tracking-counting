from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "synthetic_intersection"
VIDEO_PATH = FIXTURE_DIR / "video.mp4"
OBSERVATIONS_PATH = FIXTURE_DIR / "observations.csv"
GROUND_TRUTH_PATH = FIXTURE_DIR / "ground_truth_events.csv"
COUNTS_PATH = FIXTURE_DIR / "expected_counts.json"

WIDTH = 640
HEIGHT = 384
FPS = 10
FRAME_COUNT = 48
LINE_COLOR = (70, 220, 70)
TEXT_COLOR = (240, 240, 240)
BACKGROUND = (24, 24, 24)
LANE_COLOR = (50, 50, 50)
HORIZONTAL_LINE = ((0, 240), (640, 240))
VERTICAL_LINE = ((400, 384), (400, 0))


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    class_id: int
    label: str
    color: tuple[int, int, int]
    start_frame: int
    end_frame: int
    start_center: tuple[float, float]
    end_center: tuple[float, float]
    size: tuple[int, int]

    def active(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame

    def center(self, frame_index: int) -> tuple[float, float]:
        progress = (frame_index - self.start_frame) / (self.end_frame - self.start_frame)
        x = self.start_center[0] + progress * (self.end_center[0] - self.start_center[0])
        y = self.start_center[1] + progress * (self.end_center[1] - self.start_center[1])
        return x, y


TRAJECTORIES = [
    Trajectory(
        track_id=1,
        class_id=2,
        label="down",
        color=(255, 170, 80),
        start_frame=0,
        end_frame=24,
        start_center=(140.0, 60.0),
        end_center=(140.0, 320.0),
        size=(54, 28),
    ),
    Trajectory(
        track_id=2,
        class_id=2,
        label="up",
        color=(120, 200, 255),
        start_frame=10,
        end_frame=38,
        start_center=(500.0, 330.0),
        end_center=(500.0, 80.0),
        size=(54, 28),
    ),
    Trajectory(
        track_id=3,
        class_id=7,
        label="right",
        color=(120, 255, 160),
        start_frame=4,
        end_frame=28,
        start_center=(60.0, 120.0),
        end_center=(520.0, 120.0),
        size=(74, 34),
    ),
    Trajectory(
        track_id=4,
        class_id=2,
        label="left",
        color=(220, 120, 255),
        start_frame=16,
        end_frame=42,
        start_center=(560.0, 300.0),
        end_center=(220.0, 300.0),
        size=(54, 28),
    ),
    Trajectory(
        track_id=5,
        class_id=2,
        label="distractor",
        color=(180, 180, 180),
        start_frame=0,
        end_frame=47,
        start_center=(70.0, 335.0),
        end_center=(200.0, 335.0),
        size=(44, 24),
    ),
]

GROUND_TRUTH_EVENTS = [
    {
        "frame_index": 17,
        "video_time_s": 1.7,
        "track_id": 1,
        "class_id": 2,
        "line_name": "horizontal_midline",
        "direction": "down",
    },
    {
        "frame_index": 21,
        "video_time_s": 2.1,
        "track_id": 2,
        "class_id": 2,
        "line_name": "horizontal_midline",
        "direction": "up",
    },
    {
        "frame_index": 22,
        "video_time_s": 2.2,
        "track_id": 3,
        "class_id": 7,
        "line_name": "vertical_gate",
        "direction": "right",
    },
    {
        "frame_index": 29,
        "video_time_s": 2.9,
        "track_id": 4,
        "class_id": 2,
        "line_name": "vertical_gate",
        "direction": "left",
    },
]

EXPECTED_COUNTS = {
    "horizontal_midline": {"down": 1, "up": 1},
    "vertical_gate": {"left": 1, "right": 1},
}


def _bbox(center: tuple[float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = int(round(center[0] - width / 2))
    y1 = int(round(center[1] - height / 2))
    x2 = int(round(center[0] + width / 2))
    y2 = int(round(center[1] + height / 2))
    return x1, y1, x2, y2


def _draw_background(frame: np.ndarray) -> None:
    frame[:] = BACKGROUND
    cv2.rectangle(frame, (0, 90), (WIDTH, 150), LANE_COLOR, -1)
    cv2.rectangle(frame, (0, 270), (WIDTH, 330), LANE_COLOR, -1)
    cv2.rectangle(frame, (120, 0), (180, HEIGHT), LANE_COLOR, -1)
    cv2.rectangle(frame, (460, 0), (520, HEIGHT), LANE_COLOR, -1)
    cv2.line(frame, *HORIZONTAL_LINE, LINE_COLOR, 2)
    cv2.line(frame, *VERTICAL_LINE, LINE_COLOR, 2)
    cv2.putText(
        frame,
        "horizontal_midline",
        (12, 232),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        LINE_COLOR,
        2,
    )
    cv2.putText(frame, "vertical_gate", (408, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LINE_COLOR, 2)


def _write_observations() -> None:
    with OBSERVATIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "track_id",
                "class_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        for frame_index in range(FRAME_COUNT):
            for trajectory in TRAJECTORIES:
                if not trajectory.active(frame_index):
                    continue
                center = trajectory.center(frame_index)
                x1, y1, x2, y2 = _bbox(center, trajectory.size)
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "track_id": trajectory.track_id,
                        "class_id": trajectory.class_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": round(center[0], 6),
                        "center_y": round(center[1], 6),
                    }
                )


def _write_ground_truth() -> None:
    with GROUND_TRUTH_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "video_time_s",
                "track_id",
                "class_id",
                "line_name",
                "direction",
            ],
        )
        writer.writeheader()
        for row in GROUND_TRUTH_EVENTS:
            writer.writerow(row)


def _write_counts() -> None:
    with COUNTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(EXPECTED_COUNTS, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_video() -> None:
    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    try:
        for frame_index in range(FRAME_COUNT):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            _draw_background(frame)
            for trajectory in TRAJECTORIES:
                if not trajectory.active(frame_index):
                    continue
                center = trajectory.center(frame_index)
                x1, y1, x2, y2 = _bbox(center, trajectory.size)
                cv2.rectangle(frame, (x1, y1), (x2, y2), trajectory.color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 16, 16), 2)
                cv2.putText(
                    frame,
                    f"id={trajectory.track_id}",
                    (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    TEXT_COLOR,
                    1,
                )
            cv2.putText(
                frame,
                f"frame={frame_index:02d} time={frame_index / FPS:.1f}s",
                (12, HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                TEXT_COLOR,
                2,
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_observations()
    _write_ground_truth()
    _write_counts()
    _write_video()


if __name__ == "__main__":
    main()
