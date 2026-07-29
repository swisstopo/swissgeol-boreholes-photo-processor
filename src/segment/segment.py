"""Entry point for segmenting a batch of borehole core images."""

import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from timeit import default_timer as timer

from tqdm import tqdm

from src.config import SegmentationConfig, SegmentationError
from src.mlflow_utils import (
    log_image_metadata_processed_mlflow,
    log_segmentation_results_with_mlflow,
    log_tray_segment_mlflow,
)
from src.models import (
    ImageMetadata,
    ImageMetadataProcessed,
    RulerSegmentResult,
    TraySegmentResult,
)
from src.segment.utils.core import segment_core
from src.segment.utils.ruler import ProcessRulerGroupByShape, segment_ruler
from src.segment.utils.tray import ProcessTrayGroupByShape, segment_tray

logger = logging.getLogger(__name__)


def segment_single(
    img_metadata: ImageMetadata,
    ruler_by_shape: dict[tuple[int, int, int], RulerSegmentResult],
    tray_by_shape: dict[tuple[int, int, int], TraySegmentResult],
    config: SegmentationConfig,
) -> ImageMetadataProcessed | None:
    """Segment a single image: locate its ruler, tray and core, and assemble the result.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to segment.
        ruler_by_shape (dict[tuple[int, int, int], RulerSegmentResult]): Shared ruler
            detection per image shape, computed once per shape group. Used when this
            image's shape has an entry; otherwise the ruler is detected for this image alone.
        tray_by_shape (dict[tuple[int, int, int], TraySegmentResult]): Shared tray
            detection per image shape, computed once per shape group. Used when this
            image's shape has an entry; otherwise the tray is detected for this image alone.
        config (SegmentationConfig): Tunable segmentation parameters.

    Returns:
        ImageMetadataProcessed | None: The processed image metadata with its core, tray,
        ruler detections and segmentation records, or None if segmentation failed.
    """
    detection = None

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
        )

    except SegmentationError as e:
        logger.warning("%s. Skipping.", e)
        return None

    return detection


def segment_all(
    imgs_metadata: list[ImageMetadata],
    ruler_by_shape: dict[tuple[int, int, int], RulerSegmentResult],
    tray_by_shape: dict[tuple[int, int, int], TraySegmentResult],
    config: SegmentationConfig,
) -> list[ImageMetadataProcessed]:
    """Segment every image in the batch, in parallel, reusing per-shape ruler/tray detections.

    Args:
        imgs_metadata (list[ImageMetadata]): Images to segment.
        ruler_by_shape (dict[tuple[int, int, int], RulerSegmentResult]): Shared ruler detection
            per image shape, computed once per shape group.
        tray_by_shape (dict[tuple[int, int, int], TraySegmentResult]): Shared tray detection per
            image shape, computed once per shape group.
        config (SegmentationConfig): Tunable segmentation parameters.

    Returns:
        list[ImageMetadataProcessed]: Processed image metadata for every image that segmented
        successfully. May be shorter than `imgs_metadata` if any images failed to segment.
    """
    worker = partial(
        segment_single,
        tray_by_shape=tray_by_shape,
        ruler_by_shape=ruler_by_shape,
        config=config,
    )

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        detections = [
            detection
            for detection in tqdm(
                executor.map(worker, imgs_metadata), total=len(imgs_metadata), desc="Segmenting images"
            )
            if detection is not None
        ]

    return detections


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
    t_start = timer()

    # Step 1: Try to estimate image foreground (moving part) and ruler, once per shape group
    tray_by_shape = ProcessTrayGroupByShape(config.tray_group, config.n_workers).run(imgs_metadata)
    ruler_by_shape = ProcessRulerGroupByShape(config.ruler, config.n_workers).run(imgs_metadata)

    if with_mlflow:
        for (tray_h, tray_w, _), tray_result in tray_by_shape.items():
            log_tray_segment_mlflow(
                result=tray_result,
                filename=f"segmentation_{tray_h}x{tray_w}",
                subfolder="debug",
            )

    detections = segment_all(
        imgs_metadata=imgs_metadata,
        ruler_by_shape=ruler_by_shape,
        tray_by_shape=tray_by_shape,
        config=config,
    )

    if with_mlflow:
        for detection in detections:
            log_image_metadata_processed_mlflow(
                result=detection,
                filename=f"{detection.borehole_id}",
                subfolder="debug",
            )

        log_segmentation_results_with_mlflow(detections, time=timer() - t_start)

    return detections
