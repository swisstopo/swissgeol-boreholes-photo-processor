"""Abstract orchestration for the borehole photo processing pipeline, and its two concrete pipelines."""

import contextlib
import glob
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import tifffile
from PIL import Image
from tqdm import tqdm

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
from src.segment.segment_cores import segment_cores
from src.segment.segment_cuttings import segment_cuttings
from src.stitching.config import StitchingConfig
from src.stitching.stitching_cores import StitchingBatchCores, stitching_batch_cores, stitching_cores
from src.stitching.stitching_cuttings import StitchingBatchCuttings, stitching_batch_cuttings, stitching_cuttings

M = TypeVar("M", bound=ImageMetadata)
P = TypeVar("P", bound=ImageMetadata)
Q = TypeVar("Q")


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


class PipelineRunner(ABC, Generic[M, P, Q]):
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
    def _segment(
        self, imgs_metadata: list[M], config: SegmentationConfig, with_mlflow: bool, debug: bool, cache: bool
    ) -> list[P]:
        """Segment the collected images.

        Args:
            imgs_metadata (list[M]): Images to segment.
            config (SegmentationConfig): Tunable segmentation parameters.
            with_mlflow (bool): Whether to log artifacts to MLflow.
            debug (bool): Whether to additionally log debug images to MLflow.
            cache (bool): Whether to eagerly load and cache each image's cropped region in memory.

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
    def _collate_stitch(self, imgs: list[P], config: StitchingConfig) -> list[Q]:
        """Split processed images into independent batches, ready to be stitched.

        Args:
            imgs (list[P]): Processed image metadata to stitch together.
            config (StitchingConfig): Tunable layout parameters.

        Returns:
            list[Q]: One batch per output figure, each self-contained.
        """
        ...

    @abstractmethod
    def _batch_stitch(self, batch: Q, config: StitchingConfig) -> Image.Image:
        """Render a single batch into one output figure.

        Args:
            batch (Q): One batch produced by `_collate_stitch`.
            config (StitchingConfig): Tunable layout parameters.

        Returns:
            Image.Image: The stitched output image for this batch.
        """
        ...

    def _stitch(
        self,
        batch: Q,
        prefix: str,
        config: StitchingConfig,
        output_dir: Path,
        with_mlflow: bool = False,
        run_id: str | None = None,
    ) -> None:
        """Render one batch and write/log its output image; unit of work for the stitching pool.

        Args:
            batch (Q): One batch produced by `_collate_stitch`.
            prefix (str): Output filename stem (without extension) for this batch's figure.
            config (StitchingConfig): Tunable layout parameters.
            output_dir (Path): Directory the output files are written to.
            with_mlflow (bool, optional): Whether to log this figure as an MLflow artifact. Defaults to False.
            run_id (str | None, optional): MLflow run to attach the logged artifact to. Only used
                when `with_mlflow` is True. Defaults to None.
        """
        img = self._batch_stitch(batch, config)
        if with_mlflow:
            log_artifact_with_mlflow(
                img=img,
                filename=prefix,
                run_id=run_id,
            )

        img.resize(
            (
                int(img.size[0] * config.web_downscale_factor),
                int(img.size[1] * config.web_downscale_factor),
            )
        ).save(output_dir / f"{prefix}.jpg", quality=config.web_output_quality)
        img.save(output_dir / f"{prefix}.tif")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: PipelineConfig,
        with_mlflow: bool = False,
        debug: bool = False,
        nested: bool = False,
        log_path: Path | None = None,
        cache: bool = False,
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
            cache (bool): Whether to eagerly load and cache each image's cropped region in memory.
        """
        with _mlflow_run(input_dir.name, with_mlflow=with_mlflow, nested=nested):
            imgs_metadata = self._collect(input_dir, with_mlflow=with_mlflow)

            # segmentation
            logging.info("--- Segmentation ---")
            detections = self._segment(
                imgs_metadata, config=config.segmentation, with_mlflow=with_mlflow, debug=debug, cache=cache
            )

            # evaluation of detection
            logging.info("--- Evaluation ---")
            self._evaluate(detections, config=config.evaluation, folder_name=input_dir.name, with_mlflow=with_mlflow)

            # stitching
            logging.info("--- Stitching ---")
            batches = self._collate_stitch(detections, config=config.stitching)

            # save output
            output_dir.mkdir(parents=True, exist_ok=True)
            active_run = mlflow.active_run()

            worker = partial(
                self._stitch,
                config=config.stitching,
                output_dir=output_dir,
                with_mlflow=with_mlflow,
                run_id=active_run.info.run_id if with_mlflow and active_run is not None else None,
            )

            with ThreadPoolExecutor(max_workers=config.stitching.n_workers) as ex:
                for _ in tqdm(
                    ex.map(worker, batches, [f"{input_dir.name}_{idx + 1:03d}" for idx in range(len(batches))]),
                    total=len(batches),
                    desc="Stitching images",
                ):
                    pass

            logging.info("Created %d output figure(s) in %s", len(batches), output_dir)

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
        cache: bool = False,
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
            cache (bool): Whether to eagerly load and cache each image's cropped region in memory.
        """
        with _mlflow_run(input_dir.name, with_mlflow=with_mlflow) as active_run:
            subdirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
            logging.info("Found %d folders to process in %s", len(subdirs), input_dir.name)
            for subdir in tqdm(subdirs, desc="Processing folders"):
                self.run(
                    input_dir=subdir,
                    output_dir=output_dir / subdir.name,
                    config=config,
                    with_mlflow=with_mlflow,
                    debug=debug,
                    nested=True,
                    cache=cache,
                )

            if active_run is not None:
                log_batch_evaluation_summary_csv(active_run.info.run_id)
                if log_path is not None:
                    upload_log_to_mlflow(log_path)


class CorePipelineRunner(PipelineRunner[ImageMetadataCores, ImageMetadataProcessedCores, StitchingBatchCores]):
    """Runs the core-photos pipeline: segment core/tray/ruler, evaluate, and stitch into strips."""

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[ImageMetadataCores]:
        """Collect all TIF images from the input directory and parse filename metadata."""
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
        self,
        imgs_metadata: list[ImageMetadataCores],
        config: SegmentationConfig,
        with_mlflow: bool,
        debug: bool,
        cache: bool,
    ) -> list[ImageMetadataProcessedCores]:
        """Segment core/tray/ruler regions."""
        return segment_cores(imgs_metadata, config=config, with_mlflow=with_mlflow, debug=debug, cache=cache)

    def _evaluate(
        self,
        detections: list[ImageMetadataProcessedCores],
        config: EvaluationConfig,
        folder_name: str,
        with_mlflow: bool,
    ) -> None:
        """Evaluate detections and log the results to MLflow, if enabled."""
        if with_mlflow:
            results = evaluate_detections(detections, config)
            log_evaluation_results_with_mlflow(results, folder_name=folder_name)

    def _collate_stitch(
        self, imgs: list[ImageMetadataProcessedCores], config: StitchingConfig
    ) -> list[StitchingBatchCores]:
        """Chunk cores into batches."""
        return stitching_cores(imgs, config=config)

    def _batch_stitch(self, batch: StitchingBatchCores, config: StitchingConfig) -> Image.Image:
        """Stitch one batch of cores into a canvas."""
        return stitching_batch_cores(
            cores=batch.cores,
            shared_ruler_steps=batch.shared_ruler_steps,
            shared_borehole_id=batch.shared_borehole_id,
            fallback_scale=batch.fallback_scale,
            config=config,
        )


class CuttingsPipelineRunner(
    PipelineRunner[ImageMetadataCuttings, ImageMetadataProcessedCuttings, StitchingBatchCuttings]
):
    """Runs the cuttings pipeline: segment (placeholder), and arrange into a grid.

    No evaluation step exists yet for cuttings, so `_evaluate` is left at the base class's no-op.
    """

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[ImageMetadataCuttings]:
        """Collect cuttings images from the input directory."""
        return collect_cuttings(input_dir, with_mlflow=with_mlflow)

    def _segment(
        self,
        imgs_metadata: list[ImageMetadataCuttings],
        config: SegmentationConfig,
        with_mlflow: bool,
        debug: bool,
        cache: bool,
    ) -> list[ImageMetadataProcessedCuttings]:
        """Segment cuttings (placeholder)."""
        return segment_cuttings(imgs_metadata, config=config, with_mlflow=with_mlflow, debug=debug, cache=cache)

    def _collate_stitch(
        self, imgs: list[ImageMetadataProcessedCuttings], config: StitchingConfig
    ) -> list[StitchingBatchCuttings]:
        """Chunk cuttings into pages."""
        return stitching_cuttings(imgs, config)

    def _batch_stitch(self, batch: StitchingBatchCuttings, config: StitchingConfig) -> Image.Image:
        """Stitch one page of cuttings into a canvas."""
        return stitching_batch_cuttings(
            cuttings=batch.cuttings,
            shared_borehole_id=batch.shared_borehole_id,
            config=config,
        )
