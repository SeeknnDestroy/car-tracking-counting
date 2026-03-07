from __future__ import annotations

from dataclasses import dataclass

from car_tracking_counting.counting import TrackObservation
from car_tracking_counting.io import load_observations_csv
from car_tracking_counting.tracking import TrackerBackend


@dataclass
class ObservationReplayTracker(TrackerBackend):
    observations_by_frame: dict[int, list[TrackObservation]]
    frame_index: int = 0

    @classmethod
    def from_csv(cls, observations_path: str) -> ObservationReplayTracker:
        return cls(observations_by_frame=load_observations_csv(observations_path))

    def track(self, frame: object) -> list[TrackObservation]:
        observations = self.observations_by_frame.get(self.frame_index, [])
        self.frame_index += 1
        return observations
