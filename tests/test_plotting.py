import csv
from pathlib import Path

from car_tracking_counting.plotting import generate_plots


def test_generate_plots_from_events_csv(tmp_path: Path) -> None:
    events_path = tmp_path / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
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
        writer.writerow(
            {
                "frame_index": 1,
                "video_time_s": 0.2,
                "track_id": 1,
                "class_id": 2,
                "line_name": "midline",
                "direction": "down",
            }
        )
        writer.writerow(
            {
                "frame_index": 4,
                "video_time_s": 0.8,
                "track_id": 2,
                "class_id": 7,
                "line_name": "midline",
                "direction": "up",
            }
        )

    outputs = generate_plots(events_path, tmp_path)

    assert outputs["counts_over_time"].exists()
    assert outputs["counts_by_direction"].exists()
    assert outputs["counts_over_time"].stat().st_size > 0
    assert outputs["counts_by_direction"].stat().st_size > 0
