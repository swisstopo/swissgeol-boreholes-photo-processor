"""Entry point for segmenting a batch of borehole core images."""

import logging

from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import (
    log_image_metadata_processed_mlflow,
    log_segmentation_summary_mlflow,
    log_tray_segment_mlflow,
)
from src.models import ImageMetadata, ImageMetadataProcessed, SegmentationRecord
from src.segment.utils import (
    SegmentationError,
    segment_core_from_tray,
    segment_ruler,
    segment_ruler_by_group,
    segment_tray_by_group,
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
    to per-image thresholding if unavailable). The ruler is detected on several images per
    shape group and the median-scale detection is reused for every image in that group,
    including images that fall back to per-image tray segmentation.

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
    records: list[SegmentationRecord] = []

    # Step 1: Try to estimate image foreground (moving part) and ruler, once per shape group
    tray_by_shape = segment_tray_by_group(imgs_metadata, config.tray_multiple)
    ruler_by_shape = segment_ruler_by_group(imgs_metadata, config.ruler)
    foreground_group_by_shape = {shape: idx for idx, shape in enumerate(tray_by_shape)}

    if with_mlflow:
        for (tray_h, tray_w, _), tray_result in tray_by_shape.items():
            log_tray_segment_mlflow(
                result=tray_result,
                filename=f"segment-tray-{tray_h}x{tray_w}",
                subfolder="debug",
            )

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            shape = img_metadata.shape

            # Step 2: Check if shared ruler detected, otherwise computes it
            shared_ruler = ruler_by_shape.get(shape)
            detection_ruler = shared_ruler or segment_ruler(img_metadata, config.ruler)

            # Step 3: Use the group's shared tray if available, otherwise fallback to single
            shared_tray = tray_by_shape.get(shape)
            detection_tray = shared_tray or segment_tray_single(img_metadata, config.tray_single)

            # Step 4: Trim wooden tray (top/bottom) and black background (all sides) around the core
            detection_core = segment_core_from_tray(img_metadata, detection_tray, config=config.core)

            detection = ImageMetadataProcessed.from_metadata(
                metadata=img_metadata,
                core=detection_core,
                tray=detection_tray,
                ruler=detection_ruler,
            )

            # Step 5: Save detections and records
            foreground_group = foreground_group_by_shape.get(shape)
            detections.append(detection)
            records.append(
                SegmentationRecord(
                    filename=img_metadata.image_path.name,
                    approach="foreground" if foreground_group is not None else "fallback",
                    foreground_group=foreground_group,
                )
            )

            if with_mlflow:
                log_image_metadata_processed_mlflow(
                    result=detection,
                    filename=f"{img_metadata.image_path.stem}",
                    subfolder="debug",
                )

        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)

    fallback_count = sum([record.approach == "fallback" for record in records])
    logger.info("Segmented %d/%d image(s) using the fallback (per-image) approach.", fallback_count, len(records))

    if with_mlflow:
        log_segmentation_summary_mlflow(num_foreground_groups=len(tray_by_shape), images=records)

    return detections
