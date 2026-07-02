"""Module for image segmentation."""

import logging
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.transform import rescale
from tqdm import tqdm

from src.config import SegmentationConfig
from src.mlflow_utils import log_artifact_with_mlflow
from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


def _first_below_threshold(values: np.ndarray, threshold: float, reverse: bool = False) -> int:
    """Find the first index in the array where the value is below the threshold.

    Args:
        values (np.ndarray): 1D array of values to search.
        threshold (float): Threshold value to compare against.
        reverse (bool): If True, search from the end of the array.

    Returns:
        int: The index of the first value below the threshold. If no such value is found, returns 0 if reverse is
        False, or len(values) - 1 if reverse is True.
    """
    indices = range(len(values) - 1, -1, -1) if reverse else range(len(values))
    for i in indices:
        if values[i] < threshold:
            return i
    return 0 if not reverse else len(values) - 1


def _load_image(image_path: Path) -> np.ndarray:
    """Load a TIF image and normalize it to an RGB float array in [0, 1].

    Grayscale (2D) images are converted to 3-channel by stacking. Images with
    channel counts other than 1 or 3 are passed through unmodified.
    Uses tifffile instead of PIL since raw borehole scans may be 16-bit, which
    PIL does not handle as reliably for downstream processing.

    Args:
        image_path (Path): Path to the TIF image to load.

    Returns:
        np.ndarray: RGB image array with float values in [0, 1].
    """
    img = tifffile.imread(str(image_path))

    # grayscale to RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # normalize to [0, 1]
    if img.dtype == np.uint8:
        return img.astype(float) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(float) / 65535.0
    else:
        if img.max() == 0:
            raise SegmentationError(f"Image is blank (all-zero pixels): {image_path}")
        return img.astype(float) / img.max()


def _apply_threshold_and_clean(
    img: np.ndarray, min_object_size: int, opening_disk: int, closing_disk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholding to the input image and return a binary mask and grayscale image.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be thresholded.
        min_object_size (int): Minimum size of objects to be retained.
        opening_disk (int): Size of the disk for binary opening.
        closing_disk (int): Size of the disk for binary closing.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the cleaned binary mask and the grayscale image.
    """
    # thresholding: convert to grayscale, then apply triangle thresholding
    grey = rgb2gray(img)
    thresh = threshold_triangle(grey)
    binary_mask = grey > thresh

    # morphology: remove small objects and fill holes
    cleaned = opening(binary_mask, footprint=disk(opening_disk))
    cleaned = closing(cleaned, footprint=disk(closing_disk))
    cleaned = remove_small_objects(cleaned, max_size=min_object_size - 1)

    return cleaned, grey


def _select_bbox(
    props: list,
    img_height: int,
    min_bbox_height: int,
    edge_margin_top: int,
    edge_margin_bottom: int,
    min_size_for_bottom: int,
) -> tuple[int, int, int, int]:
    """Select the bounding box of the core region from the list of region properties.

    Assumptions:
    - The core region is the largest region that does not touch the top edge of the image
    - The core region may touch the bottom edge of the image if it is large enough
    - The core region has a certain minimum height
    - Union of all candidate bboxes is used to handle fragmented cores

    Fallback:
    - If no candidate regions are found, the largest region is selected as the core region.

    Args:
        props (list): List of region properties obtained from skimage.measure.regionprops.
        img_height (int): Height of the input image.
        min_bbox_height (int): Minimum height for a candidate core bounding box.
        edge_margin_top (int): Ignore top edge of image (ruler).
        edge_margin_bottom (int): Ignore bottom edge of image (ruler).
        min_size_for_bottom (int): Minimum area for a candidate core to touch the bottom edge of the image.

    Returns:
        tuple[int, int, int, int]: A tuple containing the coordinates of the bounding box
        in the format (min_row, min_col, max_row, max_col).
    """
    candidates = [
        r
        for r in props
        if (r.bbox[2] - r.bbox[0]) > min_bbox_height  # exclude ruler
        and r.bbox[0] > edge_margin_top  # doesn't touch top edge
        and (
            r.bbox[2] <= img_height - edge_margin_bottom or r.area > min_size_for_bottom
        )  # only large regions can touch bottom
    ]
    if not candidates:
        # fallback: just pick the largest region
        candidates = [max(props, key=lambda r: r.area)]

    # union of all candidate bboxes to handle fragmented cores
    min_row = min(r.bbox[0] for r in candidates)
    min_col = min(r.bbox[1] for r in candidates)
    max_row = max(r.bbox[2] for r in candidates)
    max_col = max(r.bbox[3] for r in candidates)

    return (min_row, min_col, max_row, max_col)


def _tray_trim(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    tray_sat_threshold: float,
) -> tuple[int, int, int, int]:
    """Trim the bounding box to exclude the wooden tray based on saturation.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be trimmed.
        bbox (tuple[int, int, int, int]): Bounding box coordinates in the format (min_row, min_col, max_row, max_col).
        tray_sat_threshold (float): Saturation above this value is treated as wooden tray (not rock).

    Returns:
        tuple[int, int, int, int]: Trimmed bounding box as (left, top, right, bottom),
        matching CoreSegmentResult.bounding_box convention.
    """
    min_row, min_col, max_row, max_col = bbox
    cropped = img[min_row:max_row, min_col:max_col]

    hsv = rgb2hsv(cropped)
    saturation = hsv[:, :, 1]  # 0 = grey, 1 = vivid colour
    row_saturation = np.mean(saturation, axis=1)
    col_saturation = np.mean(saturation, axis=0)  # mean per column instead of per row

    top_trim = _first_below_threshold(row_saturation, tray_sat_threshold)
    bottom_trim = _first_below_threshold(row_saturation, tray_sat_threshold, reverse=True)
    left_trim = _first_below_threshold(col_saturation, tray_sat_threshold)
    right_trim = _first_below_threshold(col_saturation, tray_sat_threshold, reverse=True)

    return (
        min_col + left_trim,  # left
        min_row + top_trim,  # top
        min_col + right_trim,  # right
        min_row + bottom_trim,  # bottom
    )


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
        list[ImageMetadataProcessed]: A list of processed image metadata objects, one per input image.
    """
    config = config or SegmentationConfig()
    factor = config.downscale_factor
    detections: list[ImageMetadataProcessed] = []

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        try:
            img = _load_image(img_metadata.image_path)
            detect_img = rescale(img, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img

            binary, grey = _apply_threshold_and_clean(
                detect_img,
                min_object_size=max(1, round(config.min_object_size * factor**2)),  # factor**2 for area-based configs
                opening_disk=max(1, round(config.opening_disk * factor)),
                closing_disk=max(1, round(config.closing_disk * factor)),
            )
            props = regionprops(label(binary), intensity_image=grey)

            if not props:
                raise SegmentationError(f"No regions found in image {img_metadata.image_path}")

            min_row, min_col, max_row, max_col = _select_bbox(
                props,
                detect_img.shape[0],
                min_bbox_height=max(1, round(config.min_bbox_height * factor)),
                edge_margin_top=round(config.edge_margin_top * factor),
                edge_margin_bottom=round(config.edge_margin_bottom * factor),
                min_size_for_bottom=round(config.min_size_for_bottom * factor**2),
            )

            detect_bounding_box = _tray_trim(
                detect_img,
                (min_row, min_col, max_row, max_col),
                tray_sat_threshold=config.tray_sat_threshold,
            )

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

            detections.append(
                ImageMetadataProcessed.from_metadata(
                    metadata=img_metadata,
                    result=CoreSegmentResult(bounding_box=bounding_box),
                )
            )
        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)

    return detections
