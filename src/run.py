"""CLI entry point for the borehole photo processing pipeline."""

import datetime
import logging
from pathlib import Path

import click

from src.config import PipelineConfig
from src.pipeline_runner import CorePipelineRunner, CuttingsPipelineRunner, PipelineRunner


def _run(runner: PipelineRunner, input_dir: Path, output_dir: Path, mlflow: bool, debug: bool, config: Path) -> None:
    """Run the given pipeline runner over an already-validated set of CLI options.

    Args:
        runner (PipelineRunner): The pipeline (core or cuttings) to run.
        input_dir (Path): Path to the input directory.
        output_dir (Path): Path to the output directory.
        mlflow (bool): Whether to log artifacts to MLflow.
        debug (bool): Whether to log debug images to MLflow.
        config (Path): Path to the YAML config file for segmentation and stitching parameters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{input_dir.name}_{timestamp}.log"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    pipeline_config = PipelineConfig.from_yaml(config)

    has_subdirs = any(p.is_dir() for p in input_dir.iterdir())
    run_fn = runner.batch_run if has_subdirs else runner.run
    run_fn(
        input_dir=input_dir,
        output_dir=output_dir,
        config=pipeline_config,
        with_mlflow=mlflow,
        debug=debug,
        log_path=log_path if mlflow else None,
    )


def _pipeline_options(f):
    """Attach the CLI options shared by the cores and cuttings commands."""
    f = click.option(
        "--config",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=Path("config.yaml"),
        show_default=True,
        help="Path to the YAML config file for segmentation and stitching parameters.",
    )(f)
    f = click.option("--debug", is_flag=True, help="Whether to log debug images to MLflow.")(f)
    f = click.option("--mlflow", is_flag=True, help="Whether to log artifacts to MLflow.")(f)
    f = click.option(
        "--output",
        "output_dir",
        type=click.Path(file_okay=False, path_type=Path),
        required=True,
        help="Path to the output directory.",
    )(f)
    f = click.option(
        "--input",
        "input_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True,
        help="Path to the input directory.",
    )(f)
    return f


@click.group()
def cli() -> None:
    """Process borehole photos from input to output directory."""


@cli.command("cores")
@_pipeline_options
def cores_command(input_dir: Path, output_dir: Path, mlflow: bool, debug: bool, config: Path) -> None:
    """Run the core-photos pipeline."""
    _run(CorePipelineRunner(), input_dir, output_dir, mlflow, debug, config)


@cli.command("cuttings")
@_pipeline_options
def cuttings_command(input_dir: Path, output_dir: Path, mlflow: bool, debug: bool, config: Path) -> None:
    """Run the cuttings pipeline."""
    _run(CuttingsPipelineRunner(), input_dir, output_dir, mlflow, debug, config)


def main_cores() -> None:
    """CLI entry point for the core-photos pipeline."""
    cores_command()


def main_cuttings() -> None:
    """CLI entry point for the cuttings pipeline."""
    cuttings_command()


if __name__ == "__main__":
    cli()
