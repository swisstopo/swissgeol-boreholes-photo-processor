"""Core bbox detection: trimming the wooden tray and black background around the core."""

import logging
from itertools import groupby
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny
from skimage.filters import prewitt_h, prewitt_v
from skimage.transform import hough_line, hough_line_peaks

from src.config import SegmentationCoreTrimConfig
from src.models import CoreSegmentResult, ImageMetadataCores, ImageSegmentResult
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def _is_in_range_ratio(
    values: np.ndarray, threshold_low: float = 0.0, threshold_high: float = 1.0, ratio: float = 0.0
) -> bool:
    """Check whether the fraction of values within (threshold_low, threshold_high) exceeds ratio.

    Args:
        values (np.ndarray): Per-pixel value map to threshold (e.g. an HSV saturation or value channel).
        threshold_low (float): Lower bound of the range values must fall strictly above.
        threshold_high (float): Upper bound of the range values must fall strictly below.
        ratio (float): Fraction of in-range pixels required for the check to pass.

    Returns:
        bool: True if the fraction of pixels within (threshold_low, threshold_high) is greater than ratio.
    """
    im_thresh = (threshold_low < values) & (values < threshold_high)
    return im_thresh.mean() > ratio


def _compute_row_stats(values: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    """Classify each row of a per-pixel value map as valid or invalid (tray/background).

    A row is invalid if the fraction of its pixels above `threshold` reaches `ratio`. Pass a
    transposed value map to classify columns instead (e.g. to trim left/right).

    Args:
        values (np.ndarray): 2D per-pixel value map (e.g. saturation channel) to threshold.
        threshold (float): Value above which a pixel is considered tray/background.
        ratio (float): Fraction of tray/background pixels in a row required to classify the row as invalid.

    Returns:
        np.ndarray: Boolean mask, True for rows classified as valid.
    """
    confs_row = (values > threshold).mean(axis=1)
    return confs_row < ratio


def _find_valid_intervals(detections: np.ndarray) -> list[tuple[int, int]]:
    """Group valid row/column indices into contiguous intervals.

    Args:
        detections (np.ndarray): Sorted indices of rows/columns classified as valid.

    Returns:
        list[tuple[int, int]]: Start and end index of each interval, sorted by length in
            descending order. Single-index (empty) intervals are dropped.
    """
    # Group detections to intervals
    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]
    results = np.array([[g[0], g[-1]] for g in groups])
    results = [tuple(x) for x in sorted(results.tolist(), key=lambda x: x[1] - x[0], reverse=True)]

    # Remove empty intervals [x, x]
    return [result for result in results if result[0] != result[1]]


def _find_horizontal_lines(img_gray: np.ndarray, config: SegmentationCoreTrimConfig) -> list[int]:
    """Detect y-coordinates of prominent horizontal lines (e.g. tray dividers) in the image.

    Runs a Hough transform on the horizontal-edge map (Prewitt) restricted to near-horizontal
    lines, keeping only the strongest peaks. Used to split the tray into bands for top/bottom
    trimming.

    Args:
        img_gray (np.ndarray): Grayscale image of the cropped tray region.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[int]: Row indices of the detected horizontal lines, sorted ascending, bracketed
            by `0` and `img_gray.shape[0]`.
    """
    min_distance = int(config.downscale_factor * config.min_line_hough_interval)

    h, theta, d = hough_line(np.abs(prewitt_h(img_gray)) > config.min_line_edge_value, theta=np.array([np.pi / 2]))
    y_lines = sorted(
        [
            int(dist * np.sin(angle))
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance), strict=True)
        ]
    )
    # Remove line too close from border
    y_lines = [y_line for y_line in y_lines if y_line - min_distance > 0 and y_line + min_distance < img_gray.shape[0]]
    return [0] + y_lines + [img_gray.shape[0]]


def _find_left_right_intervals(img_hsv: np.ndarray, config: SegmentationCoreTrimConfig) -> list[tuple[int, int]]:
    """Find the column intervals to keep after trimming black background on the left/right.

    Falls back to the full image width if no interval survives the minimum height filter.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Start and end column indices of each surviving left/right
            interval, sorted by length in descending order.
    """
    # Get all segments that are valid and drop short ones
    detections = np.nonzero(
        _compute_row_stats(
            values=-img_hsv[:, :, 2].T, threshold=-config.background_val_threshold, ratio=config.background_val_vratio
        )
    )[0]

    lr_trims = [
        lr_trim
        for lr_trim in _find_valid_intervals(detections=detections)
        if config.downscale_factor * config.min_segment_height_px <= lr_trim[1] - lr_trim[0]
    ]

    if len(lr_trims) == 0:
        lr_trims = [(0, img_hsv.shape[1] - 1)]

    return lr_trims


def _find_top_bottom_intervals(
    img_hsv: np.ndarray,
    img_gray: np.ndarray,
    y_lines: list[int],
    lr_trims: list[tuple[int, int]],
    config: SegmentationCoreTrimConfig,
) -> list[tuple[int, int]]:
    """Find the row intervals to keep after trimming wooden tray and black background top/bottom.

    Splits the tray into row bands at the detected horizontal divider lines (`y_lines`), then
    classifies each band as tray/background if it is mostly wood-colored (hue, saturation, and
    low edge texture) or mostly dark background (value channel), restricted to the surviving
    left/right columns. Surviving (non-tray, non-background) bands are merged into candidate
    intervals, ranked by a score favoring intervals that are both large and close to the
    vertical center. Falls back to the full image height if fewer than two rows survive.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        img_gray (np.ndarray): Gray image of the cropped tray region.
        y_lines (list[int]): Row indices of detected horizontal dividers, sorted ascending.
        lr_trims (list[tuple[int, int]]): Start and end column indices of the surviving
            left/right intervals, used to restrict the columns considered.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Start and end row indices of each candidate top/bottom
            interval, sorted by score in descending order.
    """
    im_canny = np.abs(prewitt_v(canny(img_gray, sigma=config.wood_texture_sigma)))
    detections = []
    for start, end in zip(y_lines[:-1], y_lines[1:], strict=True):
        im_segment = np.concatenate([img_hsv[start:end, lr[0] : lr[1] + 1, :] for lr in lr_trims], axis=1)
        im_canny_segment = np.concatenate([im_canny[start:end, lr[0] : lr[1] + 1] for lr in lr_trims], axis=1)
        im_hue, im_sat, im_val = np.moveaxis(im_segment, 2, 0)

        is_wood_texture = im_canny_segment.mean() < config.wood_texture_ratio

        is_wood_hue = _is_in_range_ratio(
            values=im_hue,
            threshold_low=config.wood_hue_threshold_low,
            threshold_high=config.wood_hue_threshold_high,
            ratio=config.wood_hratio,
        )

        is_wood_sat = _is_in_range_ratio(
            values=im_sat,
            threshold_low=config.wood_sat_threshold_low,
            threshold_high=config.wood_sat_threshold_high,
            ratio=config.wood_hratio,
        )

        is_background = _is_in_range_ratio(
            values=im_val,
            threshold_high=config.background_val_threshold,
            ratio=config.background_val_hratio,
        )

        if not (is_wood_sat and is_wood_hue and is_wood_texture) and not is_background:
            detections.extend(list(range(start, end)))

    if len(detections) <= 1:
        detections = list(range(0, y_lines[-1]))

    tb_intersects = _find_valid_intervals(np.array(detections))

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
    img_metadata: ImageMetadataCores, tray: ImageSegmentResult, config: SegmentationCoreTrimConfig
) -> CoreSegmentResult:
    """Trim the bounding box to exclude the wooden tray and black background around the core.

    Trims left/right based on the value (brightness) channel to drop black background. Trims
    top/bottom by splitting the tray into row bands at Hough-detected horizontal divider lines,
    then keeping the bands that don't match a wooden-tray profile (saturation, hue, and edge
    texture) and aren't black background.

    Args:
        img_metadata (ImageMetadataCores): Metadata of the image to load and trim.
        tray (ImageSegmentResult): Bounding box to trim, in the original image's coordinate space.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        CoreSegmentResult: bbox is the trimmed bounding box as (left, top, right, bottom), in the
            original image's coordinate space. bbox_segments holds one bbox per surviving
            left/right sub-segment. y_lines holds the detected horizontal divider row positions,
            for debug visualization.
    """
    t_start = timer()

    img = img_metadata.load_image(factor=config.downscale_factor)
    x_min, y_min, x_max, y_max = scale_bbox(tray.bbox, factor=config.downscale_factor)

    hsv = rgb2hsv(img[round(y_min) : round(y_max + 1), round(x_min) : round(x_max + 1)])
    gray = rgb2gray(img[round(y_min) : round(y_max + 1), round(x_min) : round(x_max + 1)])

    # Remove black background (left/right)
    lr_trims = _find_left_right_intervals(img_hsv=hsv, config=config)
    left_trim_background = np.array(lr_trims)[:, 0].min().item()
    right_trim_background = np.array(lr_trims)[:, 1].max().item()

    # Remove wood / background (top/bottom). Pick the best-scored interval (large and
    # close to vertical center)
    y_lines = _find_horizontal_lines(img_gray=gray, config=config)
    top_trim, bottom_trim = _find_top_bottom_intervals(
        img_hsv=hsv,
        img_gray=gray,
        y_lines=y_lines,
        lr_trims=lr_trims,
        config=config,
    )[0]

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
        y_lines=((y_min + np.array(y_lines)) / config.downscale_factor).tolist(),
    )
