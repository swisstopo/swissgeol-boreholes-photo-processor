"""Helper functions for image segmentation."""

import logging
from itertools import groupby
from pathlib import Path

import numpy as np
import tifffile
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gaussian, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.transform import rescale
from sklearn.mixture import GaussianMixture
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


def _estimate_foreground(
    imgs: list[ImageMetadata],
    factor: float = 0.125,
    sigma: float = 5,
    n_min: int = 10,
) -> np.ndarray | None:
    """Estimate the foreground mask shared by a batch of images taken from a static camera position.

    Loads and downscales every image, then computes the per-pixel standard deviation across the
    stack: pixels that change between shots (the core) get a high value, while the static
    background (tray, backdrop) stays low.

    Args:
        imgs (list[ImageMetadata]): Batch of images assumed to share the same camera position.
        factor (float): Downscale factor applied to each image before comparison.
        sigma (float): Gaussian blur applied to each image beforehand.
        n_min (int): Minimum number of successfully loaded images required to estimate a
            foreground. Below this, there isn't enough data for a reliable per-pixel std.

    Returns:
        np.ndarray | None: Foreground weight map normalized to [0, 1], or None.
    """
    imgs_stack: list[np.ndarray] = []
    img_shape: tuple[int, ...] | None = None
    for img_metadata in tqdm(imgs, desc="Load images"):
        try:
            img_ = _load_image(img_metadata.image_path)
            img_scale = rescale(img_, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img_
        except SegmentationError as e:
            logger.warning("%s. Skipping.", e)
            continue

        if img_shape is None:
            img_shape = img_scale.shape
        elif img_shape != img_scale.shape:
            logger.warning("Inconsistent image sizes")
            return None

        img_gray_ = rgb2gray(img_scale)
        img_blur_ = gaussian(img_gray_, sigma=sigma)
        imgs_stack.append(img_blur_)

    # At least n_min images for statistics
    if len(imgs_stack) <= n_min:
        return None

    # Compute STD between images to highlight changes in background
    return np.stack(imgs_stack).std(axis=0)


def _estimate_foreground_bbox(foreground: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Fit the foreground distribution and derive a bounding box for the core region.

    Assumes the foreground shows the highest variance. Fits a 2-component GMM over the
    per-pixel values and selects the component with the highest mean as foreground. The
    foreground mask threshold is set to mu - std of that component, and the largest
    connected region in the resulting mask is taken as the core.

    Args:
        foreground (np.ndarray | None): Foreground distribution map or None if unavailable.

    Returns:
        tuple[int, int, int, int] | None: Bounding box as (x_min, y_min, x_max, y_max), or None.
    """
    if foreground is None:
        return None

    # Fit GMM to get background and foreground distributions
    gmm = GaussianMixture(n_components=2)
    gmm.fit(foreground.flatten().reshape(-1, 1))
    means = np.asarray(gmm.means_)
    covariances = np.asarray(gmm.covariances_)
    id_foreground = np.argmax(means)
    foreground_map = foreground > means[id_foreground] - np.sqrt(covariances[id_foreground])

    # Foreground is defined as the largest connected region (area)
    props = regionprops(label(foreground_map), intensity_image=foreground)

    if not props:
        return None

    props = sorted(props, key=lambda x: x.area, reverse=True)
    bbox = props[0].bbox
    return (bbox[1], bbox[0], bbox[3], bbox[2])


def _apply_threshold_and_clean(
    img: np.ndarray,
    min_object_size: int,
    opening_disk: int,
    closing_disk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholding to the input image and return a cleaned binary mask.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be thresholded.
        min_object_size (int): Minimum size of objects to be retained.
        opening_disk (int): Size of the disk for binary opening.
        closing_disk (int): Size of the disk for binary closing.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cleaned binary mask, and the grayscale image it
        was derived from.
    """
    # Look for optimal threshold
    img_gray = rgb2gray(img)
    thresh = threshold_triangle(img_gray)

    # morphology: smooth region boundaries and remove small objects
    cleaned = opening(img_gray > thresh, footprint=disk(opening_disk))
    cleaned = closing(cleaned, footprint=disk(closing_disk))
    cleaned = remove_small_objects(cleaned, max_size=min_object_size - 1)

    return cleaned, img_gray


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

    return (min_col, min_row, max_col - 1, max_row - 1)


def _find_non_tray_interval(values: np.ndarray, threshold: float, ratio: float) -> tuple[int, int]:
    """Find the largest contiguous row interval that is not tray, based on a per-pixel value map.

    A row is classified as tray if the fraction of its pixels above `threshold` reaches
    `ratio`. The largest run of consecutive non-tray rows is returned.

    Args:
        values (np.ndarray): 2D per-pixel value map (e.g. saturation channel) to threshold.
        threshold (float): Value above which a pixel is considered tray.
        ratio (float): Fraction of tray pixels in a row required to classify the row as tray.

    Returns:
        tuple[int, int]: Start and end row indices (exclusive of tray) of the largest
            non-tray interval.
    """
    confs_row = (values > threshold).mean(axis=1)

    # Detect transition in sequence 1: True to False, -1 : False to True
    detections = np.nonzero(confs_row < ratio)[0]

    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]
    result = np.array([[g[0], g[-1]] for g in groups])

    # Best interval as largest interval
    id_best = np.argmax(result[:, 1] - result[:, 0])

    return result[id_best, 0].item(), result[id_best, 1].item()


def _tray_trim(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    tray_sat_threshold: float,
    tray_sat_ratio: float,
) -> tuple[int, int, int, int]:
    """Trim the bounding box to exclude the wooden tray based on saturation.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be trimmed.
        bbox (tuple[int, int, int, int]): Bounding box coordinates in the format (min_row, min_col, max_row, max_col).
        tray_sat_threshold (float): Saturation above this value is treated as wooden tray (not rock).
        tray_sat_ratio (float): Fraction of tray-saturation pixels in a row required to
            classify the row as tray.

    Returns:
        tuple[int, int, int, int]: Trimmed bounding box as (left, top, right, bottom),
        matching CoreSegmentResult.bounding_box convention.
    """
    x_min, y_min, x_max, y_max = bbox

    hsv = rgb2hsv(img[y_min : y_max + 1, x_min : x_max + 1])
    top_trim, bottom_trim = _find_non_tray_interval(hsv[:, :, 1], tray_sat_threshold, tray_sat_ratio)

    return (
        x_min,  # left
        y_min + top_trim,  # top
        x_max,  # right
        y_min + bottom_trim,  # bottom
    )
