"""Module for image segmentation."""

import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gaussian, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.transform import rescale
from tqdm import tqdm

from src.models import ImageMetadata

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


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


def _estimate_foreground(imgs: list[ImageMetadata], factor: float = 0.125, sigma: float = 5) -> np.ndarray | None:
    """Estimate the foreground mask shared by a batch of images taken from a static camera position.

    Loads and downscales every image, then computes the per-pixel standard deviation across the
    stack: pixels that change between shots (the core) get a high value, while the static
    background (tray, backdrop) stays low.

    Args:
        imgs (list[ImageMetadata]): Batch of images assumed to share the same camera position.
        factor (float): Downscale factor applied to each image before comparison.
        sigma (float): Standard deviation of the Gaussian blur applied to each image beforehand.

    Returns:
        np.ndarray | None: Foreground weight map normalized to [0, 1], or None if the images
        don't share a common shape after downscaling.
    """
    imgs_stack: list[np.ndarray] = []
    for img_metadata in tqdm(imgs, desc="Load images"):
        try:
            img_ = _load_image(img_metadata.image_path)
            img_scale = rescale(img_, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img_
            img_gray_ = rgb2gray(img_scale)
            img_blur_ = gaussian(img_gray_, sigma=sigma)
            imgs_stack.append(img_blur_)
        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)

    if not all(im.shape == imgs_stack[0].shape for im in imgs_stack[1:]):
        logger.warning("Disrcapencies in image shapes")
        return None

    # Compute STD between images to highlight changes in background
    img_stack = np.stack(imgs_stack).std(axis=0)
    return img_stack / (img_stack.max() + 1e-16)


def _compute_core_features(
    img: np.ndarray,
    foreground: np.ndarray | None,
) -> np.ndarray:
    """Compute a grayscale feature map used for thresholding, weighted by the foreground estimate.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]).
        foreground (np.ndarray | None): Foreground weight map from `_estimate_foreground`, or None
        if unavailable, in which case the plain grayscale image is used.

    Returns:
        np.ndarray: Grayscale feature map, optionally weighted by the foreground estimate.
    """
    img_gray = rgb2gray(img)
    return img_gray * foreground if foreground is not None else img_gray


def _apply_threshold_and_clean(
    img: np.ndarray,
    min_object_size: int,
    opening_disk: int,
    closing_disk: int,
) -> np.ndarray:
    """Apply thresholding to the input image and return a cleaned binary mask.

    Args:
        img (np.ndarray): Grayscale feature map (float, [0, 1]) to be thresholded.
        min_object_size (int): Minimum size of objects to be retained.
        opening_disk (int): Size of the disk for binary opening.
        closing_disk (int): Size of the disk for binary closing.

    Returns:
        np.ndarray: The cleaned binary mask.
    """
    # Look for optimal threshold
    thresh = threshold_triangle(img)
    binary_mask = img > thresh

    # morphology: remove small objects and fill holes
    cleaned = opening(binary_mask, footprint=disk(opening_disk))
    cleaned = closing(cleaned, footprint=disk(closing_disk))
    cleaned = remove_small_objects(cleaned, max_size=min_object_size - 1)

    return cleaned


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

    # forward/reverse are intentionally asymmetric: the reverse trim stops on the last tray-free
    # pixel rather than one past it, so a tray-coloured pixel can never survive into the crop,
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


def _select_bbox(
    img_mask: np.ndarray,
    img_intensity: np.ndarray,
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
        img_mask (np.ndarray): Binary mask of candidate core regions.
        img_intensity (np.ndarray): Grayscale intensity image used to weight region properties.
        img_height (int): Height of the input image.
        min_bbox_height (int): Minimum height for a candidate core bounding box.
        edge_margin_top (int): Ignore top edge of image (ruler).
        edge_margin_bottom (int): Ignore bottom edge of image (ruler).
        min_size_for_bottom (int): Minimum area for a candidate core to touch the bottom edge of the image.

    Returns:
        tuple[int, int, int, int]: A tuple containing the coordinates of the bounding box
        in the format (min_row, min_col, max_row, max_col).
    """
    props = regionprops(label(img_mask), intensity_image=img_intensity)
    if not props:
        raise SegmentationError("No regions found in image")

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
