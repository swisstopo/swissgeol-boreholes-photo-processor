"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import datetime
import logging
from pathlib import Path

from src.config import PipelineConfig
from src.pipeline_runner import CorePipelineRunner, CuttingsPipelineRunner, PipelineRunner


def _main(runner: PipelineRunner) -> None:
    """Parse CLI arguments and run the given pipeline runner.

    Args:
        runner (PipelineRunner): The pipeline (core or cuttings) to run.
    """
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--mlflow", action="store_true", help="Whether to log artifacts to MLflow.")
    parser.add_argument("--debug", action="store_true", help="Whether to log debug images to MLflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file for segmentation and stitching parameters (default: config.yaml).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        parser.error(f"Input path is not a directory: {args.input}")
    if not args.config.exists():
        parser.error(f"Config file does not exist: {args.config}")

    args.output.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{args.input.name}_{timestamp}.log"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    config = PipelineConfig.from_yaml(args.config)

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        runner.batch_run(
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            with_mlflow=args.mlflow,
            debug=args.debug,
            log_path=log_path if args.mlflow else None,
        )
    else:
        runner.run(
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            with_mlflow=args.mlflow,
            debug=args.debug,
            log_path=log_path if args.mlflow else None,
        )


def main_cores() -> None:
    """CLI entry point for the core-photos pipeline."""
    _main(CorePipelineRunner())


def main_cuttings() -> None:
    """CLI entry point for the cuttings pipeline."""
    _main(CuttingsPipelineRunner())


if __name__ == "__main__":
    main_cores()
