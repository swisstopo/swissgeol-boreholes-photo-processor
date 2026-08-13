"""Abstract orchestration for the borehole photo processing pipeline, and its two concrete pipelines."""

import contextlib
import glob
import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import tifffile
from PIL import Image

from src.config import PipelineConfig, SegmentationError
from src.evaluations.config import EvaluationConfig
from src.evaluations.core import evaluate_detections
from src.mlflow_utils import (
    log_artifact_with_mlflow,
    log_batch_evaluation_summary_csv,
    log_evaluation_results_with_mlflow,
    upload_log_to_mlflow,
)
from src.models import (
    ImageMetadata,
    ImageMetadataCores,
    ImageMetadataCuttings,
    ImageMetadataProcessedCores,
    ImageMetadataProcessedCuttings,
)
from src.preprocessing.cuttings import collect_cuttings
from src.segment.config import SegmentationConfig
from src.segment.segment_cores import segment
from src.segment.segment_cuttings import segment_cuttings
from src.stitching.config import StitchingConfig
from src.stitching.stitching_cores import stitching
from src.stitching.stitching_cuttings import stitching_cuttings

M = TypeVar("M", bound=ImageMetadata)
P = TypeVar("P", bound=ImageMetadata)


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


class PipelineRunner(ABC, Generic[M, P]):
    """Template-method orchestrator shared by the core and cuttings pipelines.

    Subclasses implement `_collect`, `_segment`, and `_stitch`; `_evaluate` is a no-op unless
    overridden (only the cores pipeline currently evaluates detections). `run()` and
    `batch_run()` orchestrate the shared lifecycle and are not meant to be overridden.

    Type Parameters:
        M: Metadata type returned by `_collect` and consumed by `_segment`.
        P: Processed metadata type returned by `_segment`, consumed by `_evaluate`/`_stitch`.
    """

    @abstractmethod
    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[M]:
        """Collect and parse image metadata from input_dir.

        Args:
            input_dir (Path): Path to the directory containing raw photos.
            with_mlflow (bool): Whether to log collection stats to MLflow.

        Returns:
            list[M]: Parsed metadata for every successfully-collected image.
        """
        ...

    @abstractmethod
    def _segment(self, imgs_metadata: list[M], config: SegmentationConfig, with_mlflow: bool, debug: bool) -> list[P]:
        """Segment the collected images.

        Args:
            imgs_metadata (list[M]): Images to segment.
            config (SegmentationConfig): Tunable segmentation parameters.
            with_mlflow (bool): Whether to log artifacts to MLflow.
            debug (bool): Whether to additionally log debug images to MLflow.

        Returns:
            list[P]: Processed image metadata for every image that segmented successfully.
        """
        ...

    def _evaluate(self, detections: list[P], config: EvaluationConfig, folder_name: str, with_mlflow: bool) -> None:
        """Evaluate detection quality (e.g. log to MLflow). No-op by default.

        Args:
            detections (list[P]): Processed image metadata to evaluate.
            config (EvaluationConfig): Tunable evaluation parameters.
            folder_name (str): Name of the folder being processed, used for logging.
            with_mlflow (bool): Whether to log evaluation results to MLflow.
        """
        return None

    @abstractmethod
    def _stitch(self, imgs: list[P], config: StitchingConfig) -> Generator[Image.Image, None, None]:
        """Arrange the processed images into output figure(s).

        Args:
            imgs (list[P]): Processed image metadata to stitch together.
            config (StitchingConfig): Tunable layout parameters.

        Yields:
            Image.Image: One stitched output image at a time.
        """
        ...

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: PipelineConfig,
        with_mlflow: bool = False,
        debug: bool = False,
        nested: bool = False,
        log_path: Path | None = None,
    ) -> None:
        """Process raw photos from input_dir into stitched output figure(s) in output_dir.

        Args:
            input_dir (Path): Path to the directory containing raw photos for one borehole.
            output_dir (Path): Path to the directory where processed images will be written.
            config (PipelineConfig): Tunable segmentation, stitching, and evaluation parameters.
            with_mlflow (bool): Whether to log artifacts to MLflow.
            debug (bool): Whether to additionally log debug images (e.g. per-image tray/ruler
                detections) to MLflow. Only applies when with_mlflow is True.
            nested (bool): Whether to start a nested MLflow run under an existing active run.
            log_path (Path | None): If set, upload this run's log file to MLflow once processing
                completes. Only meaningful for a top-level (non-nested) run.
        """
        with _mlflow_run(input_dir.name, with_mlflow=with_mlflow, nested=nested):
            imgs_metadata = self._collect(input_dir, with_mlflow=with_mlflow)

            # segmentation
            detections = self._segment(imgs_metadata, config=config.segmentation, with_mlflow=with_mlflow, debug=debug)

            # evaluation of detection
            self._evaluate(detections, config=config.evaluation, folder_name=input_dir.name, with_mlflow=with_mlflow)

            # stitching
            images = self._stitch(detections, config=config.stitching)

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

                img.resize(
                    (
                        int(img.size[0] * config.stitching.web_downscale_factor),
                        int(img.size[1] * config.stitching.web_downscale_factor),
                    )
                ).save(output_dir / f"{stem}.jpg", quality=config.stitching.web_output_quality)
                img.save(output_dir / f"{stem}.tif")
            logging.info("Created %d output figure(s) in %s", idx + 1, output_dir)

            if with_mlflow and not nested and log_path is not None:
                upload_log_to_mlflow(log_path)

    def batch_run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: PipelineConfig,
        with_mlflow: bool = False,
        debug: bool = False,
        log_path: Path | None = None,
    ) -> None:
        """Accepts a root directory and runs the pipeline on all subdirectories.

        Args:
            input_dir (Path): Path to the root directory whose subdirectories each contain
                raw photos for one borehole.
            output_dir (Path): Path to the directory where processed images will be written.
            config (PipelineConfig): Tunable segmentation, stitching, and evaluation parameters.
            with_mlflow (bool): Whether to log artifacts to MLflow.
            debug (bool): Whether to additionally log debug images (e.g. per-image tray/ruler
                detections) to MLflow. Only applies when with_mlflow is True.
            log_path (Path | None): If set, upload the batch's log file to MLflow once processing
                completes.
        """
        with _mlflow_run(input_dir.name, with_mlflow=with_mlflow) as active_run:
            subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
            logging.info("Found %d folders to process in %s", len(subdirs), input_dir.name)
            for i, subdir in enumerate(subdirs, start=1):
                logging.info("Processing folder %d/%d: %s", i, len(subdirs), subdir.name)
                self.run(
                    input_dir=subdir,
                    output_dir=output_dir / subdir.name,
                    config=config,
                    with_mlflow=with_mlflow,
                    debug=debug,
                    nested=True,
                )

            if active_run is not None:
                log_batch_evaluation_summary_csv(active_run.info.run_id)
                if log_path is not None:
                    upload_log_to_mlflow(log_path)


class CorePipelineRunner(PipelineRunner[ImageMetadataCores, ImageMetadataProcessedCores]):
    """Runs the core-photos pipeline: segment core/tray/ruler, evaluate, and stitch into strips."""

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[ImageMetadataCores]:
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
        return imgs_metadata

    def _segment(
        self, imgs_metadata: list[ImageMetadataCores], config: SegmentationConfig, with_mlflow: bool, debug: bool
    ) -> list[ImageMetadataProcessedCores]:
        return segment(imgs_metadata, config=config, with_mlflow=with_mlflow, debug=debug)

    def _evaluate(
        self,
        detections: list[ImageMetadataProcessedCores],
        config: EvaluationConfig,
        folder_name: str,
        with_mlflow: bool,
    ) -> None:
        if with_mlflow:
            results = evaluate_detections(detections, config)
            log_evaluation_results_with_mlflow(results, folder_name=folder_name)

    def _stitch(
        self, imgs: list[ImageMetadataProcessedCores], config: StitchingConfig
    ) -> Generator[Image.Image, None, None]:
        return stitching(imgs, config=config)


class CuttingsPipelineRunner(PipelineRunner[ImageMetadataCuttings, ImageMetadataProcessedCuttings]):
    """Runs the cuttings pipeline: segment (placeholder), and arrange into a grid.

    No evaluation step exists yet for cuttings, so `_evaluate` is left at the base class's no-op.
    """

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[ImageMetadataCuttings]:
        return collect_cuttings(input_dir, with_mlflow=with_mlflow)

    def _segment(
        self, imgs_metadata: list[ImageMetadataCuttings], config: SegmentationConfig, with_mlflow: bool, debug: bool
    ) -> list[ImageMetadataProcessedCuttings]:
        return segment_cuttings(imgs_metadata, config=config, with_mlflow=with_mlflow, debug=debug)

    def _stitch(
        self, imgs: list[ImageMetadataProcessedCuttings], config: StitchingConfig
    ) -> Generator[Image.Image, None, None]:
        return stitching_cuttings(imgs, config=config)
