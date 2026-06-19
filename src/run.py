"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import contextlib
import logging
from pathlib import Path

import mlflow
from tqdm import tqdm

from src.mlflow_utils import log_artifact_with_mlflow
from src.models import ImageMetadata, ImageMetadataProcessed
from src.segment import segment
from src.stitching import (
    MAX_CORE_LENGTH_M,
    NUM_CORES_PER_IMAGE,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PADDING_HORIZONTAL,
    PADDING_VERTICAL,
    stitching,
)


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
    with_mlflow: bool = False,
    nested: bool = False,
    num_cores_per_image: int = NUM_CORES_PER_IMAGE,
    padding_vertical: int = PADDING_VERTICAL,
    padding_horizontal: int = PADDING_HORIZONTAL,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
    max_core_length_m: float = MAX_CORE_LENGTH_M,
) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        nested (bool): Whether to start a nested MLflow run under an existing active run.
        num_cores_per_image (int): Number of cores placed side by side per output sheet.
        padding_vertical (int): Top and bottom border height in pixels.
        padding_horizontal (int): Left and right border width in pixels.
        output_width (int): Output canvas width in pixels.
        output_height (int): Output canvas height in pixels.
        max_core_length_m (float): Maximum core length in metres (fills the strip height exactly).
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
        logging.info("Found %d TIF images in %s", len(imgs_metadata), input_dir.name)

        # segmentation
        detections: list[ImageMetadataProcessed] = segment(imgs_metadata, with_mlflow=with_mlflow)

        # stitching
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(
            stitching(
                detections,
                num_cores_per_image=num_cores_per_image,
                padding_vertical=padding_vertical,
                padding_horizontal=padding_horizontal,
                output_width=output_width,
                output_height=output_height,
                max_core_length_m=max_core_length_m,
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


def batch_run(
    input_dir: Path,
    output_dir: Path,
    with_mlflow: bool = False,
    num_cores_per_image: int = NUM_CORES_PER_IMAGE,
    padding_vertical: int = PADDING_VERTICAL,
    padding_horizontal: int = PADDING_HORIZONTAL,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
    max_core_length_m: float = MAX_CORE_LENGTH_M,
) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir (Path): Path to the root directory whose subdirectories each contain
            raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        num_cores_per_image (int): Number of cores placed side by side per output sheet.
        padding_vertical (int): Top and bottom border height in pixels.
        padding_horizontal (int): Left and right border width in pixels.
        output_width (int): Output canvas width in pixels.
        output_height (int): Output canvas height in pixels.
        max_core_length_m (float): Maximum core length in metres (fills the strip height exactly).
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow):
        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        logging.info("Found %d folders to process in %s", len(subdirs), input_dir.name)
        for subdir in tqdm(subdirs, desc="Processing folders"):
            run(
                input_dir=subdir,
                output_dir=output_dir / subdir.name,
                with_mlflow=with_mlflow,
                nested=True,
                num_cores_per_image=num_cores_per_image,
                padding_vertical=padding_vertical,
                padding_horizontal=padding_horizontal,
                output_width=output_width,
                output_height=output_height,
                max_core_length_m=max_core_length_m,
            )


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--mlflow", action="store_true", help="Whether to log artifacts to MLflow.")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=NUM_CORES_PER_IMAGE,
        help=f"Cores per output sheet (default: {NUM_CORES_PER_IMAGE}).",
    )
    parser.add_argument(
        "--padding-vertical",
        type=int,
        default=PADDING_VERTICAL,
        help=f"Top/bottom border in pixels (default: {PADDING_VERTICAL}).",
    )
    parser.add_argument(
        "--padding-horizontal",
        type=int,
        default=PADDING_HORIZONTAL,
        help=f"Left/right border in pixels (default: {PADDING_HORIZONTAL}).",
    )
    parser.add_argument(
        "--output-width", type=int, default=OUTPUT_WIDTH, help=f"Canvas width in pixels (default: {OUTPUT_WIDTH})."
    )
    parser.add_argument(
        "--output-height", type=int, default=OUTPUT_HEIGHT, help=f"Canvas height in pixels (default: {OUTPUT_HEIGHT})."
    )
    parser.add_argument(
        "--max-core-length",
        type=float,
        default=MAX_CORE_LENGTH_M,
        help=f"Max core length in metres (default: {MAX_CORE_LENGTH_M}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        parser.error(f"Input path is not a directory: {args.input}")

    kwargs = dict(
        num_cores_per_image=args.num_cores,
        padding_vertical=args.padding_vertical,
        padding_horizontal=args.padding_horizontal,
        output_width=args.output_width,
        output_height=args.output_height,
        max_core_length_m=args.max_core_length,
    )

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        batch_run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow, **kwargs)
    else:
        run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow, **kwargs)


if __name__ == "__main__":
    main()
