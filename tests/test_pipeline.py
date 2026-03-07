from pathlib import Path

from car_tracking_counting.config import (
    CountingConfig,
    LineConfig,
    ModelConfig,
    PipelineConfig,
    VideoConfig,
)
from car_tracking_counting.counting import TrackObservation
from car_tracking_counting.pipeline import run_pipeline


class FakeCapture:
    def __init__(
        self,
        frames: list[object],
        fps: float = 5.0,
        width: int = 640,
        height: int = 384,
    ) -> None:
        self._frames = frames
        self._index = 0
        self._fps = fps
        self._width = width
        self._height = height

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        import cv2

        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def release(self) -> None:
        return None


class FakeTracker:
    def __init__(self, observations_by_frame: list[list[TrackObservation]]) -> None:
        self._observations_by_frame = observations_by_frame
        self._index = 0

    def track(self, frame: object) -> list[TrackObservation]:
        if self._index >= len(self._observations_by_frame):
            return []
        observations = self._observations_by_frame[self._index]
        self._index += 1
        return observations


def test_run_pipeline_writes_deterministic_artifacts(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake")
    config = PipelineConfig(
        model=ModelConfig(
            model="yolo11n.pt",
            tracker="botsort.yaml",
            device="cpu",
            imgsz=(384, 640),
            conf=0.25,
            iou=0.5,
            class_ids=(2, 7),
        ),
        video=VideoConfig(input_path=input_video, output_dir=tmp_path / "run", save_video=False),
        counting=CountingConfig(
            lines=(
                LineConfig(
                    name="midline",
                    start=(0, 10),
                    end=(20, 10),
                    negative_to_positive="down",
                    positive_to_negative="up",
                ),
            ),
            start_time=None,
        ),
    )

    tracker = FakeTracker(
        [
            [TrackObservation(track_id=3, class_id=2, bbox_xyxy=(0, 0, 5, 5), centroid=(5.0, 5.0))],
            [
                TrackObservation(
                    track_id=3,
                    class_id=2,
                    bbox_xyxy=(0, 0, 5, 5),
                    centroid=(5.0, 15.0),
                )
            ],
            [TrackObservation(track_id=3, class_id=2, bbox_xyxy=(0, 0, 5, 5), centroid=(5.0, 5.0))],
        ]
    )

    summary = run_pipeline(
        config,
        tracker_backend=tracker,
        capture_factory=lambda _: FakeCapture(frames=[object(), object(), object()]),
    )

    events_csv = summary.artifacts.events_csv.read_text(encoding="utf-8")
    summary_json = summary.artifacts.summary_json.read_text(encoding="utf-8")

    assert summary.processed_frames == 3
    assert summary.counts_by_line == {"midline": {"down": 1, "up": 1}}
    assert "frame_index,video_time_s,track_id,class_id,line_name,direction" in events_csv
    assert "midline" in events_csv
    assert '"processed_frames": 3' in summary_json
    assert '"counts_by_line"' in summary_json
    assert summary.artifacts.counts_over_time_png.exists()
    assert summary.artifacts.counts_by_direction_png.exists()
