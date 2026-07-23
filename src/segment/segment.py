"""Entry point for segmenting a batch of borehole core images."""

import logging

from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import log_image_metadata_processed_mlflow, log_tray_segment_mlflow
from src.models import ImageMetadata, ImageMetadataProcessed
from src.segment.utils import (
    SegmentationError,
    segment_core_from_tray,
    segment_ruler,
    segment_tray_multiple,
    segment_tray_single,
)

logger = logging.getLogger(__name__)


def segment(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of processed image metadata objects.

    A bounding box is derived from the batch's shared foreground estimate (falling back
    to per-image thresholding if unavailable).

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.
    """
    config = config or SegmentationConfig()
    detections: list[ImageMetadataProcessed] = []

    # Step 1: Try to estimate image foreground (moving part)
    detection_tray_multi = segment_tray_multiple(imgs_metadata, config.tray_multiple)

    if with_mlflow:
        log_tray_segment_mlflow(
            result=detection_tray_multi,
            filename="segment-tray",
            subfolder="debug",
        )

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            # Step 2: Try to detect ruler on the image
            detection_ruler = segment_ruler(img_metadata, config.ruler)

            # Step 3: Check if tray already detected, otherwise fallback to single
            detection_tray = (
                detection_tray_multi
                if detection_tray_multi is not None
                else segment_tray_single(img_metadata, config.tray_single)
            )

            # Step 4: Trim wooden tray (top/bottom) and black background (all sides) around the core
            detection_core = segment_core_from_tray(img_metadata, detection_tray, config=config.core)

            detection = ImageMetadataProcessed.from_metadata(
                metadata=img_metadata,
                core=detection_core,
                tray=detection_tray,
                ruler=detection_ruler,
            )

            if with_mlflow:
                log_image_metadata_processed_mlflow(
                    result=detection,
                    filename=f"{img_metadata.image_path.stem}",
                    subfolder="debug",
                )

            detections.append(detection)

        except (SegmentationError, ValueError) as e:
            logger.warning("%s. Skipping.", e)

    return detections
