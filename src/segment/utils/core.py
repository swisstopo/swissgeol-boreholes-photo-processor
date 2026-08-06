"""Core bbox detection: trimming the wooden tray and black background around the core."""

from itertools import groupby
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2hsv

from src.config import SegmentationCoreConfig
from src.models import CoreSegmentResult, ImageMetadata, ImageSegmentResult
from src.utils import scale_bbox


def _find_valid_intervals(values: np.ndarray, threshold: float, ratio: float) -> list[tuple[int, int]]:
    """Find the contiguous row intervals that are valid, based on a per-pixel value map.

    A row is classified as invalid (tray/background) if the fraction of its pixels above
    `threshold` reaches `ratio`; consecutive valid rows are grouped into intervals. Pass a
    transposed value map to find column intervals instead (e.g. to trim left/right).

    Args:
        values (np.ndarray): 2D per-pixel value map (e.g. saturation channel) to threshold.
        threshold (float): Value above which a pixel is considered tray/background.
        ratio (float): Fraction of tray/background pixels in a row required to classify the row as invalid.

    Returns:
        list[tuple[int, int]]: Start and end row indices of each valid interval, sorted by
            length in descending order.

    Raises:
        SegmentationError: If every row is classified as tray (no non-tray interval found),
            or if the largest non-tray interval is degenerate (zero height).
    """
    confs_row = (values > threshold).mean(axis=1)
    detections = np.nonzero(confs_row < ratio)[0]

    if detections.size == 0:
        return [(0, values.shape[0] - 1)]

    # Group detections to intervals
    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]
    results = np.array([[g[0], g[-1]] for g in groups])
    results = [tuple(x) for x in sorted(results.tolist(), key=lambda x: x[1] - x[0], reverse=True)]

    # Remove empty intervals [x, x]
    results = [result for result in results if result[0] != result[1]]

    return results if results else [(0, values.shape[0] - 1)]


def _intersect_intervals(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compute the pairwise intersection of two sets of intervals.

    Every interval in `a` is intersected with every interval in `b`; pairs that do not
    overlap are dropped.

    Args:
        a (list[tuple[int, int]]): First set of (start, end) intervals.
        b (list[tuple[int, int]]): Second set of (start, end) intervals.

    Returns:
        list[tuple[int, int]]: Overlapping (start, end) interval for each intersecting pair.
    """
    result = []
    for s1, e1 in a:
        for s2, e2 in b:
            lo = max(s1, s2)
            hi = min(e1, e2)
            if lo <= hi:
                result.append((lo, hi))
    return result


def _find_left_right_intervals(img_hsv: np.ndarray, config: SegmentationCoreConfig) -> list[tuple[int, int]]:
    """Find the column intervals to keep after trimming black background on the left/right.

    Falls back to the full image width if no interval survives the minimum height filter.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        config (SegmentationCoreConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Start and end column indices of each surviving left/right
            interval, sorted by length in descending order.
    """
    # Get all segments that are valid and drop short ones
    lr_trims = _find_valid_intervals(
        values=-img_hsv[:, :, 2].T,
        threshold=-config.background_val_threshold,
        ratio=config.background_val_vratio,
    )
    lr_trims = [
        lr_trim
        for lr_trim in lr_trims
        if config.downscale_factor * config.min_segment_height_px <= lr_trim[1] - lr_trim[0]
    ]

    if len(lr_trims) == 0:
        lr_trims = [(0, img_hsv.shape[1] - 1)]

    return lr_trims


def _find_top_bottom_intervals(
    img_hsv: np.ndarray, lr_trims: list[tuple[int, int]], config: SegmentationCoreConfig
) -> list[tuple[int, int]]:
    """Find the row intervals to keep after trimming wooden tray and black background top/bottom.

    Intersects the wood-free intervals (saturation channel) with the background-free intervals
    (value channel), restricted to the surviving left/right columns, then ranks the resulting
    intervals by a score favoring intervals that are both large and close to the vertical center.
    Falls back to the full image height if no intersection survives.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        lr_trims (list[tuple[int, int]]): Start and end column indices of the surviving
            left/right intervals, used to restrict the columns considered.
        config (SegmentationCoreConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Start and end row indices of each candidate top/bottom
            interval, sorted by score in descending order.
    """
    tb_wood = _find_valid_intervals(
        values=np.concatenate([img_hsv[:, lr[0] : lr[1] + 1, 1] for lr in lr_trims], axis=1),
        threshold=config.wood_sat_threshold,
        ratio=config.wood_sat_hratio,
    )

    tb_background = _find_valid_intervals(
        values=-np.concatenate([img_hsv[:, lr[0] : lr[1] + 1, 2] for lr in lr_trims], axis=1),
        threshold=-config.background_val_threshold,
        ratio=config.background_val_hratio,
    )

    tb_intersects = _intersect_intervals(tb_wood, tb_background)

    return (
        # Look for interval that is large (x1-x0) and close to center (1 - |(x1+x0-H)/H|)
        sorted(
            tb_intersects,
            key=lambda x: (x[1] - x[0]) * (1 - abs((x[1] + x[0] - img_hsv.shape[0]) / img_hsv.shape[0])),
            reverse=True,
        )
        if tb_intersects
        else [(0, img_hsv.shape[0] - 1)]
    )


def segment_core(
    img_metadata: ImageMetadata, tray: ImageSegmentResult | None, config: SegmentationCoreConfig
) -> CoreSegmentResult | None:
    """Trim the bounding box to exclude the wooden tray and black background around the core.

    Trims left/right based on the value (brightness) channel to drop black background, and
    trims top/bottom based on both the saturation channel (wooden tray) and the value channel
    (black background), keeping the intersection of the two vertical trims.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and trim.
        tray (ImageSegmentResult | None): Bounding box to trim, in the original image's coordinate space.
        config (SegmentationCoreConfig): Tunable segmentation parameters.

    Returns:
        CoreSegmentResult | None: bbox is the trimmed bounding box as (left, top, right, bottom), in the
            original image's coordinate space. bbox_segments holds one bbox per surviving
            left/right sub-segment.

    Raises:
        SegmentationError: If every row or column is classified as tray/background (no valid
            interval found) for any of the three trim passes.
    """
    if tray is None:
        return None

    t_start = timer()
    img = img_metadata.load_image(factor=config.downscale_factor)
    x_min, y_min, x_max, y_max = scale_bbox(tray.bbox, factor=config.downscale_factor)

    hsv = rgb2hsv(img[round(y_min) : round(y_max + 1), round(x_min) : round(x_max + 1)])

    # Remove black background (left/right)
    lr_trims = _find_left_right_intervals(img_hsv=hsv, config=config)
    left_trim_background = np.array(lr_trims)[:, 0].min().item()
    right_trim_background = np.array(lr_trims)[:, 1].max().item()

    # Remove wood / background (top/bottom). Pick the best-scored interval (large and
    # close to vertical center)
    top_trim, bottom_trim = _find_top_bottom_intervals(img_hsv=hsv, lr_trims=lr_trims, config=config)[0]

    return CoreSegmentResult(
        bbox=scale_bbox(
            bbox=(
                x_min + left_trim_background,
                y_min + top_trim,
                x_min + right_trim_background,
                y_min + bottom_trim,
            ),
            factor=1 / config.downscale_factor,
        ),
        bbox_segments=[
            scale_bbox(
                bbox=(
                    x_min + left_trim,
                    y_min + top_trim,
                    x_min + right_trim,
                    y_min + bottom_trim,
                ),
                factor=1 / config.downscale_factor,
            )
            for left_trim, right_trim in lr_trims
        ],
        time=timer() - t_start,
    )
