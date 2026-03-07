from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


def _coerce_point(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must contain exactly two integer coordinates.")
    x, y = raw
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"{field_name} must contain integer coordinates.")
    return x, y


@dataclass(frozen=True)
class LineConfig:
    name: str
    start: tuple[int, int]
    end: tuple[int, int]
    negative_to_positive: str
    positive_to_negative: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Line name must be non-empty.")
        if self.start == self.end:
            raise ValueError(f"Line {self.name!r} must have distinct start and end points.")
        if not self.negative_to_positive.strip() or not self.positive_to_negative.strip():
            raise ValueError(f"Line {self.name!r} must define both direction labels.")
        if self.negative_to_positive == self.positive_to_negative:
            raise ValueError(
                f"Line {self.name!r} must use distinct labels for each crossing direction."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineConfig:
        return cls(
            name=str(data["name"]),
            start=_coerce_point(data["start"], "counting.lines.start"),
            end=_coerce_point(data["end"], "counting.lines.end"),
            negative_to_positive=str(data["negative_to_positive"]),
            positive_to_negative=str(data["positive_to_negative"]),
        )


@dataclass(frozen=True)
class ModelConfig:
    model: str
    tracker: str
    device: str
    imgsz: tuple[int, int]
    conf: float
    iou: float
    class_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.imgsz) != 2 or any(value <= 0 for value in self.imgsz):
            raise ValueError("model.imgsz must contain exactly two positive integers.")
        if not 0 < self.conf <= 1:
            raise ValueError("model.conf must be in the interval (0, 1].")
        if not 0 < self.iou <= 1:
            raise ValueError("model.iou must be in the interval (0, 1].")
        if not self.class_ids or any(
            not isinstance(class_id, int) or class_id < 0 for class_id in self.class_ids
        ):
            raise ValueError(
                "model.class_ids must contain one or more non-negative integers."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        imgsz = data.get("imgsz", [384, 640])
        if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
            raise ValueError("model.imgsz must be a list or tuple with exactly two values.")
        class_ids = data.get("class_ids", [2, 7])
        if not isinstance(class_ids, (list, tuple)):
            raise ValueError("model.class_ids must be a list or tuple.")
        return cls(
            model=str(data.get("model", "yolo11n.pt")),
            tracker=str(data.get("tracker", "botsort.yaml")),
            device=str(data.get("device", "cpu")),
            imgsz=(int(imgsz[0]), int(imgsz[1])),
            conf=float(data.get("conf", 0.25)),
            iou=float(data.get("iou", 0.5)),
            class_ids=tuple(int(class_id) for class_id in class_ids),
        )


@dataclass(frozen=True)
class VideoConfig:
    input_path: Path
    output_dir: Path
    save_video: bool

    def __post_init__(self) -> None:
        if not self.input_path.as_posix():
            raise ValueError("video.input_path must be set.")
        if not self.output_dir.as_posix():
            raise ValueError("video.output_dir must be set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoConfig:
        input_path = data.get("input_path")
        if not input_path:
            raise ValueError("video.input_path must be set.")
        output_dir = data.get("output_dir")
        if not output_dir:
            raise ValueError("video.output_dir must be set.")
        return cls(
            input_path=Path(str(input_path)),
            output_dir=Path(str(output_dir)),
            save_video=bool(data.get("save_video", True)),
        )


@dataclass(frozen=True)
class CountingConfig:
    lines: tuple[LineConfig, ...]
    start_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("counting.lines must contain at least one line definition.")
        line_names = [line.name for line in self.lines]
        if len(set(line_names)) != len(line_names):
            raise ValueError("counting.lines must use unique names.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CountingConfig:
        raw_lines = data.get("lines", [])
        if not isinstance(raw_lines, list):
            raise ValueError("counting.lines must be a list.")
        start_time_raw = data.get("start_time")
        start_time = None
        if isinstance(start_time_raw, str) and start_time_raw.strip():
            start_time = datetime.fromisoformat(start_time_raw)
        return cls(
            lines=tuple(LineConfig.from_dict(item) for item in raw_lines),
            start_time=start_time,
        )


@dataclass(frozen=True)
class PipelineConfig:
    model: ModelConfig
    video: VideoConfig
    counting: CountingConfig

    def resolved(self, base_dir: Path | None = None) -> PipelineConfig:
        base = base_dir or Path.cwd()
        return replace(
            self,
            video=replace(
                self.video,
                input_path=(base / self.video.input_path).resolve()
                if not self.video.input_path.is_absolute()
                else self.video.input_path,
                output_dir=(base / self.video.output_dir).resolve()
                if not self.video.output_dir.is_absolute()
                else self.video.output_dir,
            ),
        )

    def override_paths(
        self,
        *,
        input_path: str | None = None,
        output_dir: str | None = None,
        save_video: bool | None = None,
    ) -> PipelineConfig:
        video = self.video
        if input_path is not None:
            video = replace(video, input_path=Path(input_path))
        if output_dir is not None:
            video = replace(video, output_dir=Path(output_dir))
        if save_video is not None:
            video = replace(video, save_video=save_video)
        return replace(self, video=video)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.counting.start_time is not None:
            data["counting"]["start_time"] = self.counting.start_time.isoformat()
        data["video"]["input_path"] = str(self.video.input_path)
        data["video"]["output_dir"] = str(self.video.output_dir)
        return data


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    config = PipelineConfig(
        model=ModelConfig.from_dict(raw.get("model", {})),
        video=VideoConfig.from_dict(raw.get("video", {})),
        counting=CountingConfig.from_dict(raw.get("counting", {})),
    )
    return config.resolved(config_path.parent)
