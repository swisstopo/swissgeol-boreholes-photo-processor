"""Module for image segmentation."""

import logging

import numpy as np
from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import log_artifact_with_mlflow
from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.segment.utils import (
    SegmentationError,
    _apply_threshold_and_clean,
    _compute_core_features,
    _estimate_foreground,
    _load_image,
    _select_bbox,
)

logger = logging.getLogger(__name__)


def segment(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of processed image metadata objects.

    If the segmentation fails for a single image, it is skipped and a warning is logged.
    Therefore the output list may be shorter than the input list.

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.
    """
    config = config or SegmentationConfig()
    factor = config.downscale_factor
    detections: list[ImageMetadataProcessed] = []

    # Step 0: Try to estimate image foreground (moving part)
    foreground = _estimate_foreground(imgs=imgs_metadata, factor=factor)

    if with_mlflow and foreground is not None:
        log_artifact_with_mlflow(
            img=Image.fromarray((foreground * 255).astype(np.uint8)),
            filename=f"{imgs_metadata[0].borehole_id}-foreground",
        )

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            # Step 1: Read and rescale image
            img = _load_image(img_metadata.image_path)
            detect_img = rescale(img, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img

            # Step 2: Compute core feature
            img_features = _compute_core_features(
                detect_img,
                foreground=foreground,
            )

            # Step 3: Apply threshold and morphology
            img_mask = _apply_threshold_and_clean(
                img_features,
                min_object_size=max(1, round(config.min_object_size * factor**2)),  # factor**2 for area-based configs
                opening_disk=max(1, round(config.opening_disk * factor)),
                closing_disk=max(1, round(config.closing_disk * factor)),
            )

            # Step 4: Detect core based on thresholded image
            min_row, min_col, max_row, max_col = _select_bbox(
                img_mask,
                img_features,
                detect_img.shape[0],
                min_bbox_height=max(1, round(config.min_bbox_height * factor)),
                edge_margin_top=round(config.edge_margin_top * factor),
                edge_margin_bottom=round(config.edge_margin_bottom * factor),
                min_size_for_bottom=round(config.min_size_for_bottom * factor**2),
            )

            # detect_bounding_box = _tray_trim(
            #     detect_img,
            #     (min_row, min_col, max_row, max_col),
            #     tray_sat_threshold=config.tray_sat_threshold,
            # )
            detect_bounding_box = min_col, min_row, max_col, max_row

            # bounding box was computed on the downscaled image; rescale it back to the original resolution
            left, top, right, bottom = detect_bounding_box
            bounding_box = (
                detect_bounding_box
                if factor == 1.0
                else (round(left / factor), round(top / factor), round(right / factor), round(bottom / factor))
            )

            if with_mlflow:
                log_artifact_with_mlflow(
                    img=Image.fromarray((img * 255).astype(np.uint8)),
                    filename=f"{img_metadata.image_path.stem}",
                    bounding_box=bounding_box,
                )
                log_artifact_with_mlflow(
                    img=Image.fromarray((img_features * 255).astype(np.uint8)),
                    filename=f"{img_metadata.image_path.stem}-feature",
                )
                log_artifact_with_mlflow(
                    img=Image.fromarray((img_mask * 255).astype(np.uint8)),
                    filename=f"{img_metadata.image_path.stem}-mask",
                )

            detections.append(
                ImageMetadataProcessed.from_metadata(
                    metadata=img_metadata,
                    result=CoreSegmentResult(bounding_box=bounding_box),
                )
            )
        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)

    return detections
