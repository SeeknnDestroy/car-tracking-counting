from __future__ import annotations

import argparse
from pathlib import Path

from car_tracking_counting.config import load_config
from car_tracking_counting.pipeline import run_pipeline
from car_tracking_counting.plotting import generate_plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vehicle tracking and directional counting.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run tracking and counting on a video.")
    run_parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to a TOML config file.",
    )
    run_parser.add_argument("--input", help="Override the input video path from the config.")
    run_parser.add_argument("--output-dir", help="Override the output directory from the config.")
    run_parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable annotated video export for this run.",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate plots from an events CSV.")
    plot_parser.add_argument("--events", required=True, help="Path to an events.csv file.")
    plot_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the generated plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        config = config.override_paths(
            input_path=args.input,
            output_dir=args.output_dir,
            save_video=False if args.no_save_video else None,
        ).resolved()
        summary = run_pipeline(config)
        print(f"Processed {summary.processed_frames} frames from {summary.input_path}.")
        print(f"Artifacts written to {summary.output_dir}.")
        return 0

    if args.command == "plot":
        plots = generate_plots(Path(args.events), Path(args.output_dir))
        print(f"Wrote {plots['counts_over_time']}")
        print(f"Wrote {plots['counts_by_direction']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
