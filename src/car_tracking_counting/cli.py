from __future__ import annotations

import argparse
from pathlib import Path

from car_tracking_counting.config import load_config
from car_tracking_counting.evaluation import evaluate_event_files
from car_tracking_counting.pipeline import run_pipeline
from car_tracking_counting.plotting import generate_plots
from car_tracking_counting.replay import ObservationReplayTracker


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
        "--observations",
        help="Path to precomputed observations CSV used instead of model-backed tracking.",
    )
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

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate predicted crossing events against ground-truth events.",
    )
    eval_parser.add_argument(
        "--predicted",
        required=True,
        help="Path to a predicted events.csv file.",
    )
    eval_parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to a ground-truth events.csv file.",
    )
    eval_parser.add_argument("--output", required=True, help="Path to the evaluation JSON output.")
    eval_parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.2,
        help="Maximum event timestamp delta in seconds for a match.",
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
        tracker_backend = None
        if args.observations:
            tracker_backend = ObservationReplayTracker.from_csv(args.observations)
        summary = run_pipeline(config, tracker_backend=tracker_backend)
        print(f"Processed {summary.processed_frames} frames from {summary.input_path}.")
        print(f"Artifacts written to {summary.output_dir}.")
        return 0

    if args.command == "plot":
        plots = generate_plots(Path(args.events), Path(args.output_dir))
        print(f"Wrote {plots['counts_over_time']}")
        print(f"Wrote {plots['counts_by_direction']}")
        return 0

    if args.command == "evaluate":
        result = evaluate_event_files(
            args.predicted,
            args.ground_truth,
            tolerance_seconds=args.time_tolerance,
            output_path=args.output,
        )
        overall = result["overall"]
        print(
            "precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
            "missed={missed} extra={extra}".format(
                precision=overall["precision"],
                recall=overall["recall"],
                f1=overall["f1"],
                missed=overall["missed_events"],
                extra=overall["extra_events"],
            )
        )
        print(f"Wrote {args.output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
