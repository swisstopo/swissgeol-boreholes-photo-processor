"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import contextlib
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from PIL import Image

from src.models import ImageMetadata, ImageMetadataProcessed


def segment(imgs_metadata: list[ImageMetadata], with_mlflow: bool = False) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of detections.

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects, one per input image.
    """
    detections: list[ImageMetadataProcessed] = []

    for img_metadata in imgs_metadata:
        with Image.open(img_metadata.image_path) as img:
            detection = img.copy()  # placeholder
            w, h = img.size  # placeholder for bounding box dimensions

        if with_mlflow:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # convert to plot with bounding box around the detected object (placeholder)
                    plt.imshow(detection)
                    plt.gca().add_patch(plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor="red", facecolor="none"))

                    artifact_path = Path(tmp_dir) / f"{img_metadata.image_path.stem}.png"
                    plt.savefig(artifact_path)
                    mlflow.log_artifact(str(artifact_path))
            finally:
                plt.close()

        detections.append(
            ImageMetadataProcessed(metadata=img_metadata, detections=[detection], bounding_boxes=[(0, 0, w, h)])
        )

    return detections


def stitch(detections: list[ImageMetadataProcessed], dir_name: str, with_mlflow: bool = False) -> Image.Image:
    """Stitch the list of detections into a final image.

    Args:
        detections (list[ImageMetadataProcessed]): A list of processed image metadata objects.
        dir_name (str): Name used as the stem for the MLflow artifact filename.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        Image.Image: An image object representing the stitched result of the detections.
    """
    stitched_image: Image.Image = Image.new("RGB", (100, 100))  # placeholder for actual stitched image

    if with_mlflow:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                plt.imshow(stitched_image)
                artifact_path = Path(tmp_dir) / f"{dir_name}.png"
                plt.savefig(artifact_path)
                mlflow.log_artifact(str(artifact_path))
        finally:
            plt.close()

    return stitched_image


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


def run(input_dir: Path, output_dir: Path, with_mlflow: bool = False, nested: bool = False) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
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

        # segmentation
        detections: list[ImageMetadataProcessed] = segment(imgs_metadata, with_mlflow=with_mlflow)

        # stitching
        stitched_image = stitch(detections, dir_name=input_dir.name, with_mlflow=with_mlflow)

        # Write results to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        stitched_image.save(output_dir / f"{input_dir.name}.tif")
        stitched_image.save(output_dir / f"{input_dir.name}.png")


def batch_run(input_dir: Path, output_dir: Path, with_mlflow: bool = False) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir (Path): Path to the root directory whose subdirectories each contain
            raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        with_mlflow (bool): Whether to log artifacts to MLflow.
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow):
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                run(input_dir=subdir, output_dir=output_dir / subdir.name, with_mlflow=with_mlflow, nested=True)


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--mlflow", action="store_true", help="Whether to log artifacts to MLflow.")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        parser.error(f"Input path is not a directory: {args.input}")

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        batch_run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow)
    else:
        run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
