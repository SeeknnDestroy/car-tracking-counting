from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isclose

from car_tracking_counting.config import LineConfig

EPSILON = 1e-6


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    class_id: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid: tuple[float, float]


@dataclass(frozen=True)
class CountEvent:
    frame_index: int
    video_time_s: float
    track_id: int
    class_id: int
    line_name: str
    direction: str
    wall_time_iso: str | None = None

    def to_row(self) -> dict[str, str | int | float]:
        return {
            "frame_index": self.frame_index,
            "video_time_s": round(self.video_time_s, 6),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "line_name": self.line_name,
            "direction": self.direction,
        }


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
    return (
        min(a[0], b[0]) - EPSILON <= c[0] <= max(a[0], b[0]) + EPSILON
        and min(a[1], b[1]) - EPSILON <= c[1] <= max(a[1], b[1]) + EPSILON
    )


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (
        ((o1 > EPSILON and o2 < -EPSILON) or (o1 < -EPSILON and o2 > EPSILON))
        and ((o3 > EPSILON and o4 < -EPSILON) or (o3 < -EPSILON and o4 > EPSILON))
    ):
        return True

    for value, start, end, point in (
        (o1, p1, p2, q1),
        (o2, p1, p2, q2),
        (o3, q1, q2, p1),
        (o4, q1, q2, p2),
    ):
        if isclose(value, 0.0, abs_tol=EPSILON) and _on_segment(start, end, point):
            return True
    return False


def classify_crossing(
    previous_point: tuple[float, float],
    current_point: tuple[float, float],
    line: LineConfig,
) -> str | None:
    if previous_point == current_point:
        return None

    line_start = (float(line.start[0]), float(line.start[1]))
    line_end = (float(line.end[0]), float(line.end[1]))
    previous_side = _orientation(line_start, line_end, previous_point)
    current_side = _orientation(line_start, line_end, current_point)

    if isclose(previous_side, 0.0, abs_tol=EPSILON) or isclose(current_side, 0.0, abs_tol=EPSILON):
        return None
    if previous_side * current_side > 0:
        return None
    if not _segments_intersect(previous_point, current_point, line_start, line_end):
        return None
    if previous_side < 0 < current_side:
        return line.negative_to_positive
    if previous_side > 0 > current_side:
        return line.positive_to_negative
    return None


class CounterState:
    def __init__(self, lines: tuple[LineConfig, ...]) -> None:
        self._lines = lines
        self._previous_centroids: dict[int, tuple[float, float]] = {}
        self._deduped_crossings: set[tuple[int, str, str]] = set()
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        return {
            line_name: dict(sorted(direction_counts.items()))
            for line_name, direction_counts in sorted(self._counts.items())
        }

    def update(
        self,
        observations: list[TrackObservation],
        *,
        frame_index: int,
        video_time_s: float,
        wall_time_iso: str | None = None,
    ) -> list[CountEvent]:
        events: list[CountEvent] = []
        for observation in observations:
            previous = self._previous_centroids.get(observation.track_id)
            if previous is None:
                self._previous_centroids[observation.track_id] = observation.centroid
                continue

            for line in self._lines:
                direction = classify_crossing(previous, observation.centroid, line)
                if direction is None:
                    continue
                dedupe_key = (observation.track_id, line.name, direction)
                if dedupe_key in self._deduped_crossings:
                    continue
                self._deduped_crossings.add(dedupe_key)
                self._counts[line.name][direction] += 1
                events.append(
                    CountEvent(
                        frame_index=frame_index,
                        video_time_s=video_time_s,
                        track_id=observation.track_id,
                        class_id=observation.class_id,
                        line_name=line.name,
                        direction=direction,
                        wall_time_iso=wall_time_iso,
                    )
                )

            self._previous_centroids[observation.track_id] = observation.centroid

        return events
