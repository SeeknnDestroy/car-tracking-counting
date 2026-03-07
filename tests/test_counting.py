from car_tracking_counting.config import LineConfig
from car_tracking_counting.counting import CounterState, TrackObservation, classify_crossing


def test_classify_crossing_detects_horizontal_crossing() -> None:
    line = LineConfig(
        name="midline",
        start=(0, 10),
        end=(20, 10),
        negative_to_positive="down",
        positive_to_negative="up",
    )

    assert classify_crossing((5.0, 0.0), (5.0, 20.0), line) == "down"
    assert classify_crossing((5.0, 20.0), (5.0, 0.0), line) == "up"


def test_classify_crossing_ignores_non_intersections() -> None:
    line = LineConfig(
        name="gate",
        start=(10, 20),
        end=(10, 0),
        negative_to_positive="right",
        positive_to_negative="left",
    )

    assert classify_crossing((0.0, 0.0), (0.0, 20.0), line) is None
    assert classify_crossing((5.0, 5.0), (5.0, 6.0), line) is None


def test_counter_state_deduplicates_jitter_for_same_direction() -> None:
    line = LineConfig(
        name="gate",
        start=(10, 20),
        end=(10, 0),
        negative_to_positive="right",
        positive_to_negative="left",
    )
    counter = CounterState((line,))

    sequence = [
        TrackObservation(track_id=7, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(5.0, 5.0)),
        TrackObservation(track_id=7, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(15.0, 5.0)),
        TrackObservation(track_id=7, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(5.0, 5.0)),
        TrackObservation(track_id=7, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(15.0, 5.0)),
    ]

    observed_events = []
    for frame_index, observation in enumerate(sequence):
        observed_events.extend(
            counter.update([observation], frame_index=frame_index, video_time_s=frame_index / 10)
        )

    assert [(event.line_name, event.direction) for event in observed_events] == [
        ("gate", "right"),
        ("gate", "left"),
    ]
    assert counter.counts == {"gate": {"left": 1, "right": 1}}


def test_counter_state_handles_multiple_tracks() -> None:
    line = LineConfig(
        name="midline",
        start=(0, 10),
        end=(20, 10),
        negative_to_positive="down",
        positive_to_negative="up",
    )
    counter = CounterState((line,))

    counter.update(
        [
            TrackObservation(track_id=1, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(5.0, 5.0)),
            TrackObservation(track_id=2, class_id=7, bbox_xyxy=(0, 0, 1, 1), centroid=(7.0, 15.0)),
        ],
        frame_index=0,
        video_time_s=0.0,
    )
    events = counter.update(
        [
            TrackObservation(track_id=1, class_id=2, bbox_xyxy=(0, 0, 1, 1), centroid=(5.0, 15.0)),
            TrackObservation(track_id=2, class_id=7, bbox_xyxy=(0, 0, 1, 1), centroid=(7.0, 5.0)),
        ],
        frame_index=1,
        video_time_s=0.1,
    )

    assert sorted((event.track_id, event.direction) for event in events) == [(1, "down"), (2, "up")]
