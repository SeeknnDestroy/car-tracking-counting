import json
from pathlib import Path

import cv2

from car_tracking_counting.cli import main
from car_tracking_counting.config import load_config
from car_tracking_counting.io import load_events_csv, load_observations_csv

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "synthetic_intersection"


def test_config_profiles_load() -> None:
    profile_paths = [
        ROOT / "configs" / "profiles" / "horizontal_roadway.toml",
        ROOT / "configs" / "profiles" / "vertical_gate.toml",
        ROOT / "configs" / "profiles" / "intersection_dual_line.toml",
    ]

    loaded = [load_config(path) for path in profile_paths]

    assert loaded[0].counting.lines[0].name == "horizontal_midline"
    assert loaded[1].counting.lines[0].name == "vertical_gate"
    assert len(loaded[2].counting.lines) == 2


def test_fixture_metadata_and_observations_align() -> None:
    video_path = FIXTURE_DIR / "video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    observations_by_frame = load_observations_csv(FIXTURE_DIR / "observations.csv")

    assert frame_count == 48
    assert width == 640
    assert height == 384
    assert fps == 10.0
    assert min(observations_by_frame) == 0
    assert max(observations_by_frame) == 47
    assert len(observations_by_frame[0]) >= 2


def test_fixture_regression_matches_expected_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fixture-run"
    exit_code = main(
        [
            "run",
            "--config",
            str(ROOT / "configs" / "profiles" / "intersection_dual_line.toml"),
            "--input",
            str(FIXTURE_DIR / "video.mp4"),
            "--observations",
            str(FIXTURE_DIR / "observations.csv"),
            "--output-dir",
            str(output_dir),
            "--no-save-video",
        ]
    )

    assert exit_code == 0

    predicted_events = load_events_csv(output_dir / "events.csv")
    ground_truth_events = load_events_csv(FIXTURE_DIR / "ground_truth_events.csv")
    expected_counts = json.loads((FIXTURE_DIR / "expected_counts.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert predicted_events == ground_truth_events
    assert summary["counts_by_line"] == expected_counts


def test_fixture_cli_evaluation_is_perfect(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    exit_code = main(
        [
            "evaluate",
            "--predicted",
            str(FIXTURE_DIR / "ground_truth_events.csv"),
            "--ground-truth",
            str(FIXTURE_DIR / "ground_truth_events.csv"),
            "--output",
            str(evaluation_path),
        ]
    )

    assert exit_code == 0
    result = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["f1"] == 1.0
