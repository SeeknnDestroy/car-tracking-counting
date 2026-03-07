from pathlib import Path

from car_tracking_counting.evaluation import EventRecord, evaluate_event_files, evaluate_events


def _event(
    frame_index: int,
    video_time_s: float,
    line_name: str = "midline",
    direction: str = "down",
) -> EventRecord:
    return EventRecord(
        frame_index=frame_index,
        video_time_s=video_time_s,
        track_id=frame_index,
        class_id=2,
        line_name=line_name,
        direction=direction,
    )


def test_evaluate_events_perfect_prediction() -> None:
    events = [_event(2, 0.2), _event(8, 0.8, direction="up")]

    result = evaluate_events(events, events, tolerance_seconds=0.2)

    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["f1"] == 1.0
    assert result["overall"]["missed_events"] == 0
    assert result["overall"]["extra_events"] == 0


def test_evaluate_events_one_missed_event() -> None:
    predicted = [_event(2, 0.2)]
    truth = [_event(2, 0.2), _event(8, 0.8, direction="up")]

    result = evaluate_events(predicted, truth, tolerance_seconds=0.2)

    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 0.5
    assert result["overall"]["missed_events"] == 1
    assert len(result["missed_events"]) == 1


def test_evaluate_events_one_extra_event() -> None:
    predicted = [_event(2, 0.2), _event(8, 0.8, direction="up")]
    truth = [_event(2, 0.2)]

    result = evaluate_events(predicted, truth, tolerance_seconds=0.2)

    assert result["overall"]["precision"] == 0.5
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["extra_events"] == 1
    assert len(result["extra_events"]) == 1


def test_evaluate_events_time_tolerance_behavior() -> None:
    predicted = [_event(2, 0.32)]
    truth = [_event(2, 0.2)]

    in_tolerance = evaluate_events(predicted, truth, tolerance_seconds=0.15)
    out_of_tolerance = evaluate_events(predicted, truth, tolerance_seconds=0.05)

    assert in_tolerance["overall"]["matched_events"] == 1
    assert out_of_tolerance["overall"]["matched_events"] == 0


def test_evaluate_event_files_writes_json(tmp_path: Path) -> None:
    predicted_path = tmp_path / "predicted.csv"
    ground_truth_path = tmp_path / "ground_truth.csv"
    output_path = tmp_path / "evaluation.json"
    csv_body = (
        "frame_index,video_time_s,track_id,class_id,line_name,direction\n"
        "2,0.2,1,2,midline,down\n"
    )
    predicted_path.write_text(csv_body, encoding="utf-8")
    ground_truth_path.write_text(csv_body, encoding="utf-8")

    result = evaluate_event_files(
        predicted_path,
        ground_truth_path,
        tolerance_seconds=0.2,
        output_path=output_path,
    )

    assert result["overall"]["f1"] == 1.0
    assert output_path.exists()
