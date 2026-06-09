"""CLI entry point for the borehole photo processing pipeline."""

import argparse
from pathlib import Path

from PIL import Image

image_extensions = {".tif"}


def segment(images: list) -> list:
    """Segment the input image and return a list of detections.

    Args:
        images: A list of image file paths to be segmented.

    Returns:
        A list of detected objects in the image.
    """
    return []  # placeholder


def stitch(detections: list) -> Image.Image:
    """Stitch the list of detections into a final image.

    Args:
        detections: A list of detected objects.

    Returns:
        An image object representing the stitched result of the detections.
    """
    return Image.new("RGB", (100, 100))  # placeholder


def run(input_dir: Path, output_dir: Path) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir: Path to the directory containing raw borehole photos.
        output_dir: Path to the directory where processed images will be written.
    """
    # Collect all images from the input directory
    images: list = [str(f) for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    # segmentation
    detections: list = segment(images)

    # stitching
    stitched_image = stitch(detections)

    # Write results to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if stitched_image is not None:
        stitched_image.save(output_dir / f"{input_dir.name}.tif")
        stitched_image.save(output_dir / f"{input_dir.name}.png")


def batch_run(input_dir: Path, output_dir: Path) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir: A list of paths to directories containing raw borehole photos.
        output_dir: Path to the directory where processed images will be written.
    """
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            run(input_dir=subdir, output_dir=output_dir / subdir.name)


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    has_subdirs = any(p.is_dir() for p in args.input.iterdir())
    if has_subdirs:
        batch_run(input_dir=args.input, output_dir=args.output)
    else:
        run(input_dir=args.input, output_dir=args.output)
