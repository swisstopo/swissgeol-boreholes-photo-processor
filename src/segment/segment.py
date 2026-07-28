"""Entry point for segmenting a batch of borehole core images."""

import logging

from tqdm import tqdm

from src.config import SegmentationConfig, SegmentationError
from src.mlflow_utils import (
    log_image_metadata_processed_mlflow,
    log_segmentation_summary_mlflow,
    log_tray_segment_mlflow,
)
from src.models import ImageMetadata, ImageMetadataProcessed, SegmentationRecord
from src.segment.utils.core import segment_core
from src.segment.utils.ruler import ProcessRulerGroupByShape, segment_ruler
from src.segment.utils.tray import ProcessTrayGroupByShape, segment_tray

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

    # Step 1: Try to estimate image foreground (moving part) and ruler, once per shape group
    tray_by_shape = ProcessTrayGroupByShape(config.tray_group).run(imgs_metadata)
    ruler_by_shape = ProcessRulerGroupByShape(config.ruler).run(imgs_metadata)

    tray_group_by_shape = {shape: idx + 1 for idx, shape in enumerate(tray_by_shape)}
    ruler_group_by_shape = {shape: idx + 1 for idx, shape in enumerate(ruler_by_shape)}

    if with_mlflow:
        for (tray_h, tray_w, _), tray_result in tray_by_shape.items():
            log_tray_segment_mlflow(
                result=tray_result,
                filename=f"segment-tray-{tray_h}x{tray_w}",
                subfolder="debug",
            )

    # Worker for function
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            shape = img_metadata.shape

            # Step 2: Check if shared ruler detected, otherwise computes it
            shared_ruler = ruler_by_shape.get(shape)
            detection_ruler = shared_ruler or segment_ruler(img_metadata, config.ruler)

            # Step 3: Use the group's shared tray if available, otherwise fallback to single
            shared_tray = tray_by_shape.get(shape)
            detection_tray = shared_tray or segment_tray(img_metadata, config.tray_single)

            # Step 4: Trim wooden tray (top/bottom) and black background (all sides) around the core
            detection_core = segment_core(img_metadata, detection_tray, config=config.core)

            # Step 5: Save detections and records
            detection = ImageMetadataProcessed.from_metadata(
                metadata=img_metadata,
                core=detection_core,
                tray=detection_tray,
                ruler=detection_ruler,
                records=SegmentationRecord(
                    tray_approach="single" if shared_tray is None else "group",
                    tray_group=tray_group_by_shape.get(shape),
                    ruler_approach="single" if shared_ruler is None else "group",
                    ruler_group=ruler_group_by_shape.get(shape),
                ),
            )

            detections.append(detection)

            if with_mlflow:
                log_image_metadata_processed_mlflow(
                    result=detection,
                    filename=f"{img_metadata.image_path.stem}",
                    subfolder="debug",
                )

        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)

    if with_mlflow:
        log_segmentation_summary_mlflow(detections)

    return detections
