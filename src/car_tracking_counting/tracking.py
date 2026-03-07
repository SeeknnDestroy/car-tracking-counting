from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ultralytics import YOLO

from car_tracking_counting.config import ModelConfig
from car_tracking_counting.counting import TrackObservation


class TrackerBackend(Protocol):
    def track(self, frame: object) -> list[TrackObservation]:
        ...


@dataclass
class UltralyticsTracker:
    model_config: ModelConfig

    def __post_init__(self) -> None:
        self._model = YOLO(self.model_config.model)

    def track(self, frame: object) -> list[TrackObservation]:
        results = self._model.track(
            frame,
            persist=True,
            device=self.model_config.device,
            imgsz=self.model_config.imgsz,
            conf=self.model_config.conf,
            iou=self.model_config.iou,
            classes=list(self.model_config.class_ids),
            tracker=self.model_config.tracker,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes.id is None:
            return []

        xyxy = boxes.xyxy.int().cpu().tolist()
        xywh = boxes.xywh.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        observations: list[TrackObservation] = []
        for bbox_xyxy, bbox_xywh, track_id, class_id in zip(
            xyxy, xywh, track_ids, class_ids, strict=True
        ):
            center_x, center_y = bbox_xywh[0], bbox_xywh[1]
            observations.append(
                TrackObservation(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    bbox_xyxy=tuple(int(value) for value in bbox_xyxy),
                    centroid=(float(center_x), float(center_y)),
                )
            )
        return observations
