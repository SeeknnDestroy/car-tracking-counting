from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from car_tracking_counting.io import load_events_csv, write_summary_json


@dataclass(frozen=True)
class EventRecord:
    frame_index: int
    video_time_s: float
    track_id: int
    class_id: int
    line_name: str
    direction: str

    @property
    def key(self) -> tuple[str, str]:
        return self.line_name, self.direction

    @classmethod
    def from_row(cls, row: dict[str, str]) -> EventRecord:
        return cls(
            frame_index=int(row["frame_index"]),
            video_time_s=float(row["video_time_s"]),
            track_id=int(row["track_id"]),
            class_id=int(row["class_id"]),
            line_name=row["line_name"],
            direction=row["direction"],
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "frame_index": self.frame_index,
            "video_time_s": round(self.video_time_s, 6),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "line_name": self.line_name,
            "direction": self.direction,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _match_group(
    predicted: list[EventRecord],
    ground_truth: list[EventRecord],
    tolerance_seconds: float,
) -> tuple[list[tuple[EventRecord, EventRecord]], list[EventRecord], list[EventRecord]]:
    unmatched_predicted = predicted[:]
    matches: list[tuple[EventRecord, EventRecord]] = []
    missed: list[EventRecord] = []

    for truth_event in ground_truth:
        best_index: int | None = None
        best_delta: float | None = None
        for candidate_index, predicted_event in enumerate(unmatched_predicted):
            delta = abs(predicted_event.video_time_s - truth_event.video_time_s)
            if delta > tolerance_seconds:
                continue
            if best_delta is None or delta < best_delta:
                best_index = candidate_index
                best_delta = delta
        if best_index is None:
            missed.append(truth_event)
            continue
        matches.append((unmatched_predicted.pop(best_index), truth_event))

    return matches, unmatched_predicted, missed


def _count_by_key(events: list[EventRecord]) -> dict[str, int]:
    counts = Counter(f"{event.line_name}:{event.direction}" for event in events)
    return dict(sorted(counts.items()))


def evaluate_events(
    predicted_events: list[EventRecord],
    ground_truth_events: list[EventRecord],
    *,
    tolerance_seconds: float = 0.2,
) -> dict[str, Any]:
    predicted_groups: dict[tuple[str, str], list[EventRecord]] = defaultdict(list)
    ground_truth_groups: dict[tuple[str, str], list[EventRecord]] = defaultdict(list)
    for event in predicted_events:
        predicted_groups[event.key].append(event)
    for event in ground_truth_events:
        ground_truth_groups[event.key].append(event)

    all_keys = sorted(set(predicted_groups) | set(ground_truth_groups))
    matched_events: list[tuple[EventRecord, EventRecord]] = []
    extra_events: list[EventRecord] = []
    missed_events: list[EventRecord] = []
    per_key: dict[str, dict[str, Any]] = {}

    for key in all_keys:
        predicted_group = sorted(
            predicted_groups.get(key, []),
            key=lambda event: event.video_time_s,
        )
        ground_truth_group = sorted(
            ground_truth_groups.get(key, []), key=lambda event: event.video_time_s
        )
        matches, unmatched_predicted, unmatched_truth = _match_group(
            predicted_group,
            ground_truth_group,
            tolerance_seconds,
        )
        matched_events.extend(matches)
        extra_events.extend(unmatched_predicted)
        missed_events.extend(unmatched_truth)
        matched_count = len(matches)
        predicted_count = len(predicted_group)
        truth_count = len(ground_truth_group)
        precision = _safe_divide(matched_count, predicted_count)
        recall = _safe_divide(matched_count, truth_count)
        per_key[f"{key[0]}:{key[1]}"] = {
            "predicted_count": predicted_count,
            "ground_truth_count": truth_count,
            "matched_count": matched_count,
            "count_error": predicted_count - truth_count,
            "absolute_count_error": abs(predicted_count - truth_count),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
        }

    matched_count = len(matched_events)
    predicted_count = len(predicted_events)
    ground_truth_count = len(ground_truth_events)
    precision = _safe_divide(matched_count, predicted_count)
    recall = _safe_divide(matched_count, ground_truth_count)

    return {
        "tolerance_seconds": tolerance_seconds,
        "overall": {
            "predicted_events": predicted_count,
            "ground_truth_events": ground_truth_count,
            "matched_events": matched_count,
            "missed_events": len(missed_events),
            "extra_events": len(extra_events),
            "count_error": predicted_count - ground_truth_count,
            "absolute_count_error": abs(predicted_count - ground_truth_count),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
        },
        "counts": {
            "predicted": _count_by_key(predicted_events),
            "ground_truth": _count_by_key(ground_truth_events),
        },
        "by_line_direction": per_key,
        "missed_events": [event.to_dict() for event in missed_events],
        "extra_events": [event.to_dict() for event in extra_events],
    }


def evaluate_event_files(
    predicted_events_path: str | Path,
    ground_truth_events_path: str | Path,
    *,
    tolerance_seconds: float = 0.2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    predicted_events = [
        EventRecord.from_row(row) for row in load_events_csv(predicted_events_path)
    ]
    ground_truth_events = [
        EventRecord.from_row(row) for row in load_events_csv(ground_truth_events_path)
    ]
    result = evaluate_events(
        predicted_events,
        ground_truth_events,
        tolerance_seconds=tolerance_seconds,
    )
    if output_path is not None:
        write_summary_json(result, Path(output_path))
    return result
