"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import contextlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from PIL import Image


def segment(image_paths: list[Path], with_mlflow: bool = False) -> list:
    """Segment the input image and return a list of detections.

    Args:
        image_paths (list[Path]): A list of image file paths to be segmented.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list: A list of detected objects in the image.
    """
    detections: list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for image_path in image_paths:
            img = Image.open(image_path)
            detection = img.copy()  # placeholder

            # log artifact
            if with_mlflow:
                try:
                    # convert to plot with bounding box around the detected object (placeholder)
                    plt.imshow(detection)
                    w, h = img.size
                    plt.gca().add_patch(
                        plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor="red", facecolor="none")
                    )  # placeholder

                    artifact_path = Path(tmp_dir) / f"{image_path.stem}.png"
                    plt.savefig(artifact_path)
                    mlflow.log_artifact(str(artifact_path))
                finally:
                    plt.close()

            detections.append(detection)

    return detections


def stitch(detections: list, dir_name: str, with_mlflow: bool = False) -> Image.Image:
    """Stitch the list of detections into a final image.

    Args:
        detections (list): A list of detected objects.
        dir_name (str): The name of the directory where the stitched image will be saved.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        Image.Image: An image object representing the stitched result of the detections.
    """
    stitched_image: Image.Image = Image.new("RGB", (100, 100))  # placeholder for actual stitched image

    # log artifact
    if with_mlflow:
        plt.imshow(stitched_image)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / f"{dir_name}.png"
            plt.savefig(artifact_path)
            plt.close()
            mlflow.log_artifact(str(artifact_path))

    return stitched_image


def run(input_dir: Path, output_dir: Path, with_mlflow: bool = False) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos.
        output_dir (Path): Path to the directory where processed images will be written.
        with_mlflow (bool): Whether to log artifacts to MLflow.
    """
    # Collect all images from the input directory
    image_paths: list[Path] = [f for f in input_dir.iterdir() if f.suffix.lower() == ".tif"]

    # segmentation
    detections: list = segment(image_paths, with_mlflow=with_mlflow)

    # stitching
    stitched_image = stitch(detections, dir_name=input_dir.name, with_mlflow=with_mlflow)

    # Write results to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if stitched_image is not None:
        stitched_image.save(output_dir / f"{input_dir.name}.tif")
        stitched_image.save(output_dir / f"{input_dir.name}.png")


def batch_run(input_dir: Path, output_dir: Path, with_mlflow: bool = False) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir (Path): Path to the root directory whose subdirectories each contain raw borehole photos.
        output_dir (Path): Path to the directory where processed images will be written.
        with_mlflow (bool): Whether to log artifacts to MLflow.
    """
    parent_ctx = mlflow.start_run(run_name=input_dir.name) if with_mlflow else contextlib.nullcontext()
    with parent_ctx:
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                child_ctx = (
                    mlflow.start_run(run_name=subdir.name, nested=True) if with_mlflow else contextlib.nullcontext()
                )
                with child_ctx:
                    run(input_dir=subdir, output_dir=output_dir / subdir.name, with_mlflow=with_mlflow)


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--mlflow", action="store_true", help="Whether to log artifacts to MLflow.")
    args = parser.parse_args()

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        batch_run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow)
    else:
        ctx = mlflow.start_run(run_name=args.input.name) if args.mlflow else contextlib.nullcontext()
        with ctx:
            run(input_dir=args.input, output_dir=args.output, with_mlflow=args.mlflow)
