"""Entry point for segmenting a batch of borehole core images."""

import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import log_image_metadata_processed_mlflow
from src.models import ImageMetadata, ImageMetadataProcessed, ImageSegmentResult, RulerSegmentResult
from src.segment.utils import (
    SegmentationError,
    segment_core_from_tray,
    segment_ruler,
    segment_tray_multiple,
    segment_tray_single,
)

logger = logging.getLogger(__name__)


def segment_single(
    img_metadata: ImageMetadata,
    config: SegmentationConfig,
    detection_ruler: RulerSegmentResult | None = None,
    detection_tray: ImageSegmentResult | None = None,
    with_mlflow: bool = False,
) -> ImageMetadataProcessed | None:
    """Run the full segmentation pipeline (ruler, tray, core) on a single image.

    Detects the ruler and tray unless already provided by the caller, then trims the
    tray bbox to the core region.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and segment.
        config (SegmentationConfig): Tunable segmentation parameters for all stages.
        detection_ruler (RulerSegmentResult | None, optional): Precomputed ruler detection
            to reuse instead of running `segment_ruler`. Defaults to None.
        detection_tray (ImageSegmentResult | None, optional): Precomputed tray detection
            to reuse instead of running `segment_tray_single`. Defaults to None.
        with_mlflow (bool, optional): Whether to log the segmentation result as an MLflow
            debug artifact. Defaults to False.

    Returns:
        ImageMetadataProcessed | None: The image metadata enriched with the detected core,
            tray, and ruler regions, or None if segmentation failed for this image.
    """
    try:
        # Step 1: Try to detect ruler on the image
        detection_ruler_local = (
            detection_ruler if detection_ruler is not None else segment_ruler(img_metadata, config.ruler)
        )

        # Step 2: Check if tray already detected, otherwise fallback to single
        detection_tray_local = (
            detection_tray if detection_tray is not None else segment_tray_single(img_metadata, config.tray_single)
        )

        # Step 3: Remove wooden tray (up/down)
        detection_core_local = segment_core_from_tray(img_metadata, detection_tray_local, config=config.core)

        detection = ImageMetadataProcessed.from_metadata(
            metadata=img_metadata,
            core=detection_core_local,
            tray=detection_tray_local,
            ruler=detection_ruler_local,
        )

        if with_mlflow:
            log_image_metadata_processed_mlflow(
                result=detection,
                filename=f"{img_metadata.image_path.stem}",
                subfolder="debug",
            )

    except (SegmentationError, ValueError) as e:
        logger.warning("%s. Skipping.", e)
        return None

    return detection


def segment(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
    n_cores: int = 1,
) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of processed image metadata objects.

    A bounding box is derived from the batch's shared foreground estimate (falling back
    to per-image thresholding if unavailable).

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.
        n_cores (int): Number of worker processes used to segment images in parallel.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.
    """
    config = config or SegmentationConfig()

    # Try to estimate image foreground (moving part)
    detection_tray = segment_tray_multiple(imgs_metadata, config.tray_multiple)

    # Setup up worker with fixed / non iterable items
    worker = partial(segment_single, detection_tray=detection_tray, config=config, with_mlflow=with_mlflow)

    with ProcessPoolExecutor(n_cores) as executor:
        detections = list(
            tqdm(executor.map(worker, imgs_metadata), total=len(imgs_metadata), desc="Segmenting images")
        )

    return [detection for detection in detections if detection is not None]
