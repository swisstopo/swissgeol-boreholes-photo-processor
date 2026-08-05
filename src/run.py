"""CLI entry point for the borehole photo processing pipeline."""

import argparse
import contextlib
import datetime
import glob
import logging
from pathlib import Path

import mlflow
import tifffile
from tqdm import tqdm

from src.config import PipelineConfig, SegmentationError
from src.evaluations.core import evaluate_detections
from src.mlflow_utils import (
    log_artifact_with_mlflow,
    log_batch_evaluation_summary_csv,
    log_evaluation_results_with_mlflow,
    upload_log_to_mlflow,
)
from src.models import ImageMetadataCores
from src.preprocessing.cuttings import collect_cuttings
from src.segment.segment import segment
from src.segment.segment_cuttings import segment_cuttings
from src.stitching.stitching import stitching
from src.stitching.stitching_cuttings import stitching_cuttings


def _mlflow_run(
    run_name: str, with_mlflow: bool, nested: bool = False
) -> contextlib.AbstractContextManager[mlflow.ActiveRun | None]:
    """Context manager for MLflow run.

    Args:
        run_name (str): Name of the MLflow run.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        nested (bool): Whether to start a nested MLflow run.

    Returns:
        contextlib.AbstractContextManager[mlflow.ActiveRun | None]: A context manager for the
            MLflow run, yielding the active run when with_mlflow is True, else None.
    """
    if with_mlflow:
        return mlflow.start_run(run_name=run_name, nested=nested)
    return contextlib.nullcontext()


def run(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    with_mlflow: bool = False,
    debug: bool = False,
    nested: bool = False,
    log_path: Path | None = None,
    cuttings: bool = False,
) -> None:
    """Process borehole photos from input to output directory.

    Args:
        input_dir (Path): Path to the directory containing raw borehole photos (TIF format),
            or raw cuttings photos (JPG/BMP/TIF) when cuttings is set.
        output_dir (Path): Path to the directory where processed images will be written.
        config (PipelineConfig): Tunable segmentation and stitching parameters.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        debug (bool): Whether to additionally log debug images (e.g. per-image tray/ruler
            detections) to MLflow. Only applies when with_mlflow is True.
        nested (bool): Whether to start a nested MLflow run under an existing active run.
        log_path (Path | None): If set, upload this run's log file to MLflow once processing
            completes. Only meaningful for a top-level (non-nested) run.
        cuttings (bool): If set, skip core segmentation/evaluation and arrange the raw images
            into a cuttings grid instead.
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow, nested=nested):
        if cuttings:
            # collect all cuttings images
            imgs_metadata = collect_cuttings(input_dir, with_mlflow=with_mlflow)

            # segmentation
            detections = segment_cuttings(
                imgs_metadata, config=config.segmentation, with_mlflow=with_mlflow, debug=debug
            )

            # stitching
            images = stitching_cuttings(detections, config=config.stitching)
        else:
            # Collect all images from the input directory and parse filename metadata
            imgs_metadata: list[ImageMetadataCores] = []
            for f in map(Path, glob.glob(str(input_dir / "*"), include_hidden=False)):
                if f.suffix.lower() == ".tif":
                    try:
                        metadata = ImageMetadataCores.from_path(f)
                        _ = metadata.shape  # validate the file is readable before segmentation runs
                        imgs_metadata.append(metadata)
                    except (ValueError, SegmentationError, tifffile.TiffFileError) as e:
                        logging.warning("Skipping %s: %s", f.name, e)
            imgs_metadata.sort(key=lambda m: m.depth_start)
            logging.info("Found %d TIF images in %s", len(imgs_metadata), input_dir.name)

            # segmentation
            detections = segment(imgs_metadata, config=config.segmentation, with_mlflow=with_mlflow, debug=debug)

            # evaluation of detection
            if with_mlflow:
                results = evaluate_detections(detections, config.evaluation)
                log_evaluation_results_with_mlflow(results, folder_name=input_dir.name)

            # stitching
            images = stitching(detections, config=config.stitching)

        # save output images to output_dir and optionally log to MLflow
        output_dir.mkdir(parents=True, exist_ok=True)
        idx = -1  # guards against NameError in the logging call when detections is empty
        for idx, img in enumerate(images):
            stem = f"{input_dir.name}_{idx + 1:03d}"

            if with_mlflow:
                log_artifact_with_mlflow(
                    img=img,
                    filename=stem,
                )

            img.save(output_dir / f"{stem}.png")
            img.save(output_dir / f"{stem}.tif")
        logging.info("Created %d output figure(s) in %s", idx + 1, output_dir)

        if with_mlflow and not nested and log_path is not None:
            upload_log_to_mlflow(log_path)


def batch_run(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    with_mlflow: bool = False,
    debug: bool = False,
    log_path: Path | None = None,
    cuttings: bool = False,
) -> None:
    """Accepts a root directory and runs the pipeline on all subdirectories.

    Args:
        input_dir (Path): Path to the root directory whose subdirectories each contain
            raw borehole photos (TIF format).
        output_dir (Path): Path to the directory where processed images will be written.
        config (PipelineConfig): Tunable segmentation and stitching parameters.
        with_mlflow (bool): Whether to log artifacts to MLflow.
        debug (bool): Whether to additionally log debug images (e.g. per-image tray/ruler
            detections) to MLflow. Only applies when with_mlflow is True.
        log_path (Path | None): If set, upload the batch's log file to MLflow once processing
            completes.
        cuttings (bool): If set, skip core segmentation/evaluation and arrange the raw images
            into a cuttings grid instead.
    """
    with _mlflow_run(input_dir.name, with_mlflow=with_mlflow) as active_run:
        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        logging.info("Found %d folders to process in %s", len(subdirs), input_dir.name)
        for subdir in tqdm(subdirs, desc="Processing folders"):
            run(
                input_dir=subdir,
                output_dir=output_dir / subdir.name,
                config=config,
                with_mlflow=with_mlflow,
                debug=debug,
                nested=True,
                cuttings=cuttings,
            )

        if active_run is not None:
            log_batch_evaluation_summary_csv(active_run.info.run_id)
            if log_path is not None:
                upload_log_to_mlflow(log_path)


def main(cuttings: bool = False) -> None:
    """Parse CLI arguments and run the pipeline."""
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
        batch_run(
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            with_mlflow=args.mlflow,
            debug=args.debug,
            log_path=log_path if args.mlflow else None,
            cuttings=cuttings,
        )
    else:
        run(
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            with_mlflow=args.mlflow,
            debug=args.debug,
            log_path=log_path if args.mlflow else None,
            cuttings=cuttings,
        )


def main_cuttings() -> None:
    """CLI entry point for the cuttings pipeline."""
    main(cuttings=True)


if __name__ == "__main__":
    main()
