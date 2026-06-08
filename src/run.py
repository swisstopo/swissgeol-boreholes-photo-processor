"""CLI entry point for the borehole photo processing pipeline."""

import argparse
from pathlib import Path


def run(input: Path, output: Path) -> None:
    """Pipeline entry point."""
    pass


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Process borehole photos from input to output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    args = parser.parse_args()
    run(input=args.input, output=args.output)
