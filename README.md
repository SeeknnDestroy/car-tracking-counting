# Car Tracking and Counting

This repository tracks vehicles in video, counts directional crossings against configurable lines, and exports a reproducible run directory with structured artifacts:

- `events.csv` with one row per counted crossing
- `summary.json` with input metadata, config, runtime stats, and aggregated counts
- `counts_over_time.png` and `counts_by_direction.png`
- `annotated.mp4` when video export is enabled

The maintained path is a Python package and CLI built around Ultralytics tracking. The notebook remains in the repository as a secondary walkthrough of the original exploration.

## What This Repository Does

The pipeline:

1. loads a video and reads its real FPS and frame dimensions from OpenCV metadata
2. runs Ultralytics tracking on each frame
3. detects centroid crossings against named line segments
4. deduplicates repeated `(track_id, line, direction)` events to avoid count inflation from jitter
5. writes event-level and run-level artifacts to an output directory

The default configuration targets CPU-friendly usage with `yolo11n.pt` and `botsort.yaml`. Both are configurable.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
git clone https://github.com/SeeknnDestroy/car-tracking-counting.git
cd car-tracking-counting
uv sync --extra dev
```

Run the pipeline:

```bash
uv run python -m car_tracking_counting run \
  --config configs/default.toml \
  --input path/to/input.mp4 \
  --output-dir runs/example
```

Generate plots from an existing events file:

```bash
uv run python -m car_tracking_counting plot \
  --events runs/example/events.csv \
  --output-dir runs/example
```

## Configuration

The default configuration lives in [`configs/default.toml`](configs/default.toml).

Supported sections:

- `[model]`: detector weights, tracker, device, image size, confidence, IoU, and tracked class IDs
- `[video]`: input path, output directory, and whether to save an annotated video
- `[counting]`: optional `start_time` and one or more named line definitions

Each line definition provides:

- `name`
- `start` and `end` pixel coordinates
- `negative_to_positive` label
- `positive_to_negative` label

Direction labels are interpreted relative to the directed line from `start` to `end`. For example, a vertical gate defined bottom-to-top makes left-to-right motion map cleanly to `negative_to_positive = "right"`.

Example:

```toml
[[counting.lines]]
name = "horizontal_midline"
start = [0, 240]
end = [640, 240]
negative_to_positive = "down"
positive_to_negative = "up"
```

## Outputs

Each run writes a self-contained directory:

- `events.csv`: `frame_index`, `video_time_s`, `track_id`, `class_id`, `line_name`, `direction`
- `summary.json`: resolved config, input metadata, processed frame count, elapsed runtime, and counts by line
- `counts_over_time.png`: cumulative counts over video time
- `counts_by_direction.png`: aggregate counts by direction label
- `annotated.mp4`: annotated output video when enabled

## Validation

Lint and tests:

```bash
uv run ruff check
uv run pytest
```

The automated tests focus on line-crossing behavior, config validation, plotting, and a smoke-tested pipeline with mocked tracking outputs. They do not benchmark model quality or tracker accuracy.

## Limitations

- Accuracy depends on detector/tracker quality, camera angle, line placement, and scene density.
- The default model is intentionally lightweight for CPU usability, not maximum accuracy.
- The repository does not ship a sample video or frozen benchmark artifacts, so reproduction means reproducing the pipeline behavior on your own input rather than matching a bundled golden output.
- This repository focuses on inference-time tracking and counting. It does not include model training, dataset curation, or formal MOT evaluation.

## Notebook And Legacy Material

- [`car_tracking.ipynb`](car_tracking.ipynb) is kept as a secondary walkthrough of the original approach.
- Archived ONNX exploration code lives under [`legacy/`](legacy) and is not part of the maintained execution path.
