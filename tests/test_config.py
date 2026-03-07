from pathlib import Path

import pytest

from car_tracking_counting.config import load_config


def test_load_config_parses_start_time(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "yolo11n.pt"
tracker = "botsort.yaml"
device = "cpu"
imgsz = [384, 640]
conf = 0.25
iou = 0.5
class_ids = [2, 7]

[video]
input_path = "input.mp4"
output_dir = "runs/test"
save_video = false

[counting]
start_time = "2026-03-07T12:30:00"

[[counting.lines]]
name = "midline"
start = [0, 10]
end = [20, 10]
negative_to_positive = "down"
positive_to_negative = "up"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.video.input_path == (tmp_path / "input.mp4").resolve()
    assert config.video.output_dir == (tmp_path / "runs/test").resolve()
    assert config.counting.start_time is not None
    assert config.counting.start_time.isoformat() == "2026-03-07T12:30:00"


def test_load_config_rejects_invalid_line_definition(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "yolo11n.pt"
tracker = "botsort.yaml"
device = "cpu"
imgsz = [384, 640]
conf = 0.25
iou = 0.5
class_ids = [2, 7]

[video]
input_path = "input.mp4"
output_dir = "runs/test"
save_video = false

[counting]
start_time = ""

[[counting.lines]]
name = "bad"
start = [10, 10]
end = [10, 10]
negative_to_positive = "down"
positive_to_negative = "up"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct start and end points"):
        load_config(config_path)


def test_load_config_rejects_invalid_class_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "yolo11n.pt"
tracker = "botsort.yaml"
device = "cpu"
imgsz = [384, 640]
conf = 0.25
iou = 0.5
class_ids = ["car"]

[video]
input_path = "input.mp4"
output_dir = "runs/test"
save_video = false

[counting]
start_time = ""

[[counting.lines]]
name = "midline"
start = [0, 10]
end = [20, 10]
negative_to_positive = "down"
positive_to_negative = "up"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)
