# Synthetic Intersection Fixture

This fixture is a tiny deterministic benchmark for the repository's counting logic.

- `video.mp4` is a generated 640x384 synthetic clip at 10 FPS with 48 frames.
- `observations.csv` contains precomputed tracked boxes and centroids for each frame.
- `ground_truth_events.csv` is the canonical set of expected crossing events.
- `expected_counts.json` contains the expected aggregate counts by line and direction.

The fixture includes four true crossings and one distractor track that never crosses a counting line.

Regenerate the fixture assets from the declarative trajectory spec:

```bash
uv run python scripts/generate_synthetic_intersection_fixture.py
```

Replay the fixture through the maintained pipeline:

```bash
uv run python -m car_tracking_counting run \
  --config configs/profiles/intersection_dual_line.toml \
  --input fixtures/synthetic_intersection/video.mp4 \
  --observations fixtures/synthetic_intersection/observations.csv \
  --output-dir runs/synthetic_fixture \
  --no-save-video
```
