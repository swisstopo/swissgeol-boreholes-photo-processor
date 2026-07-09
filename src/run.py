"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import contextlib
import logging
from pathlib import Path

import mlflow
from tqdm import tqdm

from evaluations.core_length import check_core_length
from evaluations.core_width import check_core_width
from src.config import PipelineConfig
from src.mlflow_utils import log_artifact_with_mlflow, log_evaluation_results_with_mlflow
from src.models import ImageMetadata, ImageMetadataProcessed
from src.segment.segment import segment
from src.stitching import stitching


def _mlflow_run(run_name: str, with_mlflow: bool, nested: bool = False) -> contextlib.AbstractContextManager:
    """Context manager for MLflow run.

    Args:
        run_name (str): Name of the MLflow run.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        nested (bool): Whether to start a nested MLflow run.

    Returns:
        contextlib.AbstractContextManager: A context manager for the MLflow run.
    """
    if with_mlflow:
        return mlflow.start_run(run_name=run_name, nested=nested)
    return contextlib.nullcontext()


def run(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    with_mlflow: bool = False,
    nested: bool = False,
) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        config (PipelineConfig): Tunable segmentation and stitching parameters.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        nested (bool): Whether to start a nested MLflow run under an existing active run.
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow, nested=nested):
        # Collect all images from the input directory and parse filename metadata
        imgs_metadata: list[ImageMetadata] = []
        for f in input_dir.iterdir():
            if f.suffix.lower() == ".tif":
                try:
                    imgs_metadata.append(ImageMetadata.from_path(f))
                except ValueError as e:
                    logging.warning("Skipping %s: %s", f.name, e)
        imgs_metadata.sort(key=lambda m: m.depth_start)
        logging.info("Found %d TIF images in %s", len(imgs_metadata), input_dir.name)

        # segmentation
        detections: list[ImageMetadataProcessed] = segment(
            imgs_metadata, config=config.segmentation, with_mlflow=with_mlflow
        )

        # evaluation of detection
        if with_mlflow:
            log_evaluation_results_with_mlflow(
                results=check_core_width(detections, config.evaluation.core_width),
                filename="core_width",
            )
            log_evaluation_results_with_mlflow(
                results=check_core_length(detections, config.evaluation.core_length),
                filename="core_length",
            )

        # stitching
        output_dir.mkdir(parents=True, exist_ok=True)
        idx = -1  # guards against NameError in the logging call when detections is empty
        for idx, img in enumerate(
            stitching(
                detections,
                num_cores_per_image=config.stitching.num_cores_per_image,
                padding_vertical=config.stitching.padding_vertical,
                padding_horizontal=config.stitching.padding_horizontal,
                output_width=config.stitching.output_width,
                output_height=config.stitching.output_height,
                max_core_length_m=config.stitching.max_core_length_m,
            )
        ):
            stem = f"{input_dir.name}_{idx + 1:03d}"

            if with_mlflow:
                log_artifact_with_mlflow(
                    img=img,
                    filename=stem,
                )

            img.save(output_dir / f"{stem}.tif")
            img.save(output_dir / f"{stem}.png")
        logging.info("Created %d output figure(s) in %s", idx + 1, output_dir)


def batch_run(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    with_mlflow: bool = False,
) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir (Path): Path to the root directory whose subdirectories each contain
            raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        config (PipelineConfig): Tunable segmentation and stitching parameters.
        with_mlflow (bool): Whether to log artifacts to MLflow.
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow):
        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        logging.info("Found %d folders to process in %s", len(subdirs), input_dir.name)
        for subdir in tqdm(subdirs, desc="Processing folders"):
            run(
                input_dir=subdir,
                output_dir=output_dir / subdir.name,
                config=config,
                with_mlflow=with_mlflow,
                nested=True,
            )


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--mlflow", action="store_true", help="Whether to log artifacts to MLflow.")
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

    config = PipelineConfig.from_yaml(args.config)

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        batch_run(input_dir=args.input, output_dir=args.output, config=config, with_mlflow=args.mlflow)
    else:
        run(input_dir=args.input, output_dir=args.output, config=config, with_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
