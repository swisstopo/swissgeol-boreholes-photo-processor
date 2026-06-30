"""Module for image segmentation."""

import logging

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from tqdm import tqdm

from src.mlflow_utils import log_artifact_with_mlflow
from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


OPENING_DISK = 20  # radius for binary_opening (removes noise)
CLOSING_DISK = 20  # radius for binary_closing (fills gaps)
MIN_OBJECT_SIZE = 500  # minimum blob size in pixels
EDGE_MARGIN_TOP = 100  # ignore top edge of image (ruler)
EDGE_MARGIN_BOTTOM = 5  # ignore bottom edge of image (ruler)
MIN_BBOX_HEIGHT = 500
TRAY_SAT_THRESHOLD = 0.28  # saturation above this = wooden tray (not rock)


def _apply_threshold_and_clean(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    grey = rgb2gray(img)
    thresh = threshold_triangle(grey)
    binary_mask = grey > thresh

    return binary_mask, grey


def _select_bbox(props: list, img_height: int) -> tuple[int, int, int, int]:
    candidates = [
        r
        for r in props
        if (r.bbox[2] - r.bbox[0]) > MIN_BBOX_HEIGHT  # exclude ruler
        and r.bbox[0] > EDGE_MARGIN_TOP  # doesn't touch top edge
        and (r.bbox[2] <= img_height - EDGE_MARGIN_BOTTOM or r.area > 500_000)  # only large regions can touch bottom
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


def _tray_trim(img: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    min_row_s, min_col_s, max_row_s, max_col_s = bbox
    img_array = np.array(img)
    cropped = img_array[min_row_s:max_row_s, min_col_s:max_col_s]

    hsv = rgb2hsv(cropped)
    saturation = hsv[:, :, 1]  # 0 = grey, 1 = vivid colour
    row_saturation = np.mean(saturation, axis=1)

    # trim bottom: scan from bottom upward, find first row below threshold
    bottom_trim = len(row_saturation) - 1
    for i in range(len(row_saturation) - 1, -1, -1):
        if row_saturation[i] < TRAY_SAT_THRESHOLD:
            bottom_trim = i
            break

    # trim top: first row below threshold from top
    top_trim = 0
    for i in range(len(row_saturation)):
        if row_saturation[i] < TRAY_SAT_THRESHOLD:
            top_trim = i
            break

    # trim right and left: scan inward from each side
    col_saturation = np.mean(saturation, axis=0)  # mean per column instead of per row

    left_trim = 0
    right_trim = len(col_saturation) - 1

    for i in range(len(col_saturation)):
        if col_saturation[i] < TRAY_SAT_THRESHOLD:
            left_trim = i
            break

    for i in range(len(col_saturation) - 1, -1, -1):
        if col_saturation[i] < TRAY_SAT_THRESHOLD:
            right_trim = i
            break

    return (
        min_col_s + left_trim,  # left
        min_row_s + top_trim,  # top
        min_col_s + right_trim,  # right
        min_row_s + bottom_trim,  # bottom
    )


def segment(imgs_metadata: list[ImageMetadata], with_mlflow: bool = False) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of detections.

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects, one per input image.
    """
    detections: list[ImageMetadataProcessed] = []

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            with Image.open(img_metadata.image_path) as img:
                binary, grey = _apply_threshold_and_clean(img)
                props = regionprops(label(binary), intensity_image=grey)

                if not props:
                    raise SegmentationError(f"No regions found in image {img_metadata.image_path}")

                min_row, min_col, max_row, max_col = _select_bbox(props, img.height)

                bounding_box = _tray_trim(img, (min_row, min_col, max_row, max_col))

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


# TODO: add docstring and comments
# TODO: add tests
