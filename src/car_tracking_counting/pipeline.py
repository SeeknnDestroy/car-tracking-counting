from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2

from car_tracking_counting.config import PipelineConfig
from car_tracking_counting.counting import CounterState, CountEvent, TrackObservation
from car_tracking_counting.io import ensure_output_dir, write_events_csv, write_summary_json
from car_tracking_counting.plotting import generate_plots
from car_tracking_counting.tracking import TrackerBackend, UltralyticsTracker


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class RunArtifacts:
    events_csv: Path
    summary_json: Path
    counts_over_time_png: Path
    counts_by_direction_png: Path
    annotated_video: Path | None


@dataclass(frozen=True)
class RunSummary:
    input_path: Path
    output_dir: Path
    video_metadata: VideoMetadata
    processed_frames: int
    elapsed_seconds: float
    counts_by_line: dict[str, dict[str, int]]
    artifacts: RunArtifacts

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        payload["artifacts"] = {
            "events_csv": str(self.artifacts.events_csv),
            "summary_json": str(self.artifacts.summary_json),
            "counts_over_time_png": str(self.artifacts.counts_over_time_png),
            "counts_by_direction_png": str(self.artifacts.counts_by_direction_png),
            "annotated_video": (
                str(self.artifacts.annotated_video) if self.artifacts.annotated_video else None
            ),
        }
        return payload


def _build_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise ValueError("Video FPS metadata is missing or invalid.")
    if width <= 0 or height <= 0:
        raise ValueError("Video resolution metadata is missing or invalid.")
    return VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)


def _draw_overlay(
    frame: Any,
    observations: list[TrackObservation],
    lines: tuple[Any, ...],
    counts_by_line: dict[str, dict[str, int]],
    video_time_s: float,
) -> Any:
    for line in lines:
        cv2.line(frame, line.start, line.end, (0, 255, 0), 2)
        label_anchor = (line.start[0] + 6, line.start[1] + 24)
        cv2.putText(frame, line.name, label_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for observation in observations:
        x1, y1, x2, y2 = observation.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID {observation.track_id}",
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    y_offset = 28
    for line_name, direction_counts in sorted(counts_by_line.items()):
        label = ", ".join(
            f"{direction}={count}" for direction, count in sorted(direction_counts.items())
        )
        cv2.putText(
            frame,
            f"{line_name}: {label or '0'}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y_offset += 26

    cv2.putText(
        frame,
        f"time={video_time_s:.2f}s",
        (10, max(frame.shape[0] - 12, y_offset + 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return frame


def run_pipeline(
    config: PipelineConfig,
    *,
    tracker_backend: TrackerBackend | None = None,
    capture_factory: Callable[[str], cv2.VideoCapture] | None = None,
) -> RunSummary:
    resolved = config.resolved()
    if not resolved.video.input_path.exists():
        raise FileNotFoundError(f"Input video not found: {resolved.video.input_path}")

    capture_factory = capture_factory or cv2.VideoCapture
    capture = capture_factory(str(resolved.video.input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {resolved.video.input_path}")

    metadata = _read_video_metadata(capture)
    output_dir = ensure_output_dir(resolved.video.output_dir)
    tracker = tracker_backend or UltralyticsTracker(resolved.model)
    counter_state = CounterState(resolved.counting.lines)
    events: list[CountEvent] = []
    frame_index = 0
    video_writer: cv2.VideoWriter | None = None
    annotated_video_path: Path | None = None
    if resolved.video.save_video:
        annotated_video_path = output_dir / "annotated.mp4"
        video_writer = _build_video_writer(
            annotated_video_path,
            metadata.width,
            metadata.height,
            metadata.fps,
        )

    started_at = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            observations = tracker.track(frame)
            video_time_s = frame_index / metadata.fps
            wall_time_iso = None
            if resolved.counting.start_time is not None:
                wall_time_iso = (
                    resolved.counting.start_time + timedelta(seconds=video_time_s)
                ).isoformat()
            frame_events = counter_state.update(
                observations,
                frame_index=frame_index,
                video_time_s=video_time_s,
                wall_time_iso=wall_time_iso,
            )
            events.extend(frame_events)

            if video_writer is not None:
                rendered = _draw_overlay(
                    frame.copy(),
                    observations,
                    resolved.counting.lines,
                    counter_state.counts,
                    video_time_s,
                )
                video_writer.write(rendered)

            frame_index += 1
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    events_csv = write_events_csv(events, output_dir / "events.csv")
    plot_paths = generate_plots(events_csv, output_dir)
    summary = RunSummary(
        input_path=resolved.video.input_path,
        output_dir=output_dir,
        video_metadata=metadata,
        processed_frames=frame_index,
        elapsed_seconds=round(time.perf_counter() - started_at, 4),
        counts_by_line=counter_state.counts,
        artifacts=RunArtifacts(
            events_csv=events_csv,
            summary_json=output_dir / "summary.json",
            counts_over_time_png=plot_paths["counts_over_time"],
            counts_by_direction_png=plot_paths["counts_by_direction"],
            annotated_video=annotated_video_path,
        ),
    )
    summary_payload = summary.to_dict()
    summary_payload["config"] = resolved.to_dict()
    write_summary_json(summary_payload, summary.artifacts.summary_json)
    return summary
