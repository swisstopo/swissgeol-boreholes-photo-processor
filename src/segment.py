"""Module for image segmentation."""

import logging

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import log_artifact_with_mlflow
from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


def _apply_threshold_and_clean(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholding to the input image and return a binary mask and grayscale image.

    Args:
        img (Image.Image): Input image to be thresholded.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the binary mask and the grayscale image.
    """
    grey = rgb2gray(img)
    thresh = threshold_triangle(grey)
    binary_mask = grey > thresh

    return binary_mask, grey


def _select_bbox(
    props: list,
    img_height: int,
    min_bbox_height: int,
    edge_margin_top: int,
    edge_margin_bottom: int,
) -> tuple[int, int, int, int]:
    """Select the bounding box of the core region from the list of region properties.

    Assumptions:
    - The core region is the largest region that does not touch the top edge of the image
    - The core region may touch the bottom edge of the image if it is large enough
    - The core region has a ceratin minimum height
    - Union of all candidate bboxes is used to handle fragmented cores

    Fallback:
    - If no candidate regions are found, the largest region is selected as the core region.

    Args:
        props (list): List of region properties obtained from skimage.measure.regionprops.
        img_height (int): Height of the input image.
        min_bbox_height (int): Minimum height for a candidate core bounding box.
        edge_margin_top (int): Ignore top edge of image (ruler).
        edge_margin_bottom (int): Ignore bottom edge of image (ruler).

    Returns:
        tuple[int, int, int, int]: A tuple containing the coordinates of the bounding box
        in the format (min_row, min_col, max_row, max_col).
    """
    candidates = [
        r
        for r in props
        if (r.bbox[2] - r.bbox[0]) > min_bbox_height  # exclude ruler
        and r.bbox[0] > edge_margin_top  # doesn't touch top edge
        and (r.bbox[2] <= img_height - edge_margin_bottom or r.area > 500_000)  # only large regions can touch bottom
    ]
    if not candidates:
        # fallback: just pick the largest region
        core_region = max(props, key=lambda r: r.area)
        min_row_s = core_region.bbox[0]
        min_col_s = core_region.bbox[1]
        max_row_s = core_region.bbox[2]
        max_col_s = core_region.bbox[3]
    else:
        # union of all candidate bboxes to handles fragmented cores
        min_row_s = min(r.bbox[0] for r in candidates)
        min_col_s = min(r.bbox[1] for r in candidates)
        max_row_s = max(r.bbox[2] for r in candidates)
        max_col_s = max(r.bbox[3] for r in candidates)

    return (min_row_s, min_col_s, max_row_s, max_col_s)


def _tray_trim(
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    tray_sat_threshold: float,
) -> tuple[int, int, int, int]:
    """Trim the bounding box to exclude the wooden tray based on saturation.

    Args:
        img (Image.Image): Input image to be trimmed.
        bbox (tuple[int, int, int, int]): Bounding box coordinates in the format (min_row, min_col, max_row, max_col).
        tray_sat_threshold (float): Saturation above this value is treated as wooden tray (not rock).

    Returns:
        tuple[int, int, int, int]: Trimmed bounding box coordinates in the format (
        min_col, min_row, max_col, max_row).
    """
    min_row_s, min_col_s, max_row_s, max_col_s = bbox
    img_array = np.array(img)
    cropped = img_array[min_row_s:max_row_s, min_col_s:max_col_s]

    hsv = rgb2hsv(cropped)
    saturation = hsv[:, :, 1]  # 0 = grey, 1 = vivid colour
    row_saturation = np.mean(saturation, axis=1)

    # trim bottom: scan from bottom upward, find first row below threshold
    bottom_trim = len(row_saturation) - 1
    for i in range(len(row_saturation) - 1, -1, -1):
        if row_saturation[i] < tray_sat_threshold:
            bottom_trim = i
            break

    # trim top: first row below threshold from top
    top_trim = 0
    for i in range(len(row_saturation)):
        if row_saturation[i] < tray_sat_threshold:
            top_trim = i
            break

    # trim right and left: scan inward from each side
    col_saturation = np.mean(saturation, axis=0)  # mean per column instead of per row

    left_trim = 0
    right_trim = len(col_saturation) - 1

    for i in range(len(col_saturation)):
        if col_saturation[i] < tray_sat_threshold:
            left_trim = i
            break

    for i in range(len(col_saturation) - 1, -1, -1):
        if col_saturation[i] < tray_sat_threshold:
            right_trim = i
            break

    return (
        min_col_s + left_trim,  # left
        min_row_s + top_trim,  # top
        min_col_s + right_trim,  # right
        min_row_s + bottom_trim,  # bottom
    )


def segment(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of processed image metadata objects.

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects, one per input image.
    """
    config = config or SegmentationConfig()
    detections: list[ImageMetadataProcessed] = []

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            with Image.open(img_metadata.image_path) as img:
                binary, grey = _apply_threshold_and_clean(img)
                props = regionprops(label(binary), intensity_image=grey)

                if not props:
                    raise SegmentationError(f"No regions found in image {img_metadata.image_path}")

                min_row, min_col, max_row, max_col = _select_bbox(
                    props,
                    img.height,
                    min_bbox_height=config.min_bbox_height,
                    edge_margin_top=config.edge_margin_top,
                    edge_margin_bottom=config.edge_margin_bottom,
                )

                bounding_box = _tray_trim(
                    img,
                    (min_row, min_col, max_row, max_col),
                    tray_sat_threshold=config.tray_sat_threshold,
                )

                if with_mlflow:
                    log_artifact_with_mlflow(
                        img=img,
                        filename=f"{img_metadata.image_path.stem}",
                        bounding_box=bounding_box,
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


# TODO: add tests
# TODO: add sanity checks
