"""CLI entry point for the borehole photo processing pipeline."""

import argparse
from pathlib import Path


def run(input_dir: Path, output_dir: Path) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos.
        output_dir (Path): Path to the directory where processed images will be written.
    """
    pass


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    args = parser.parse_args()
    run(input_dir=args.input, output_dir=args.output)
