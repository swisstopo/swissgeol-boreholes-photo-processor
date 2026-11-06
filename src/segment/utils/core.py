"""Core bbox detection: trimming the wooden tray and black background around the core."""

import logging
from itertools import groupby
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import prewitt_h, scharr
from skimage.transform import hough_line, hough_line_peaks

from src.config import SegmentationCoreTrimConfig
from src.models import CoreSegmentResult, ImageMetadataCores, ImageSegmentResult
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def _in_range(values: np.ndarray, threshold_low: float = 0.0, threshold_high: float = 1.0) -> np.ndarray:
    """Boolean mask of values strictly within (threshold_low, threshold_high).

    Args:
        values (np.ndarray): Per-pixel value map to threshold (e.g. an HSV saturation or value channel).
        threshold_low (float): Lower bound of the range values must fall strictly above.
        threshold_high (float): Upper bound of the range values must fall strictly below.

    Returns:
        np.ndarray: Boolean mask, same shape as `values`.
    """
    return (threshold_low < values) & (values < threshold_high)


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
    return _in_range(values, threshold_low, threshold_high).mean() > ratio


def _find_valid_intervals(detections: np.ndarray) -> list[tuple[int, int]]:
    """Group valid row/column indices into contiguous intervals.

    Args:
        detections (np.ndarray): Sorted indices of rows/columns classified as valid.

    Returns:
        list[tuple[int, int]]: Start and end index of each interval. Single-index (empty) intervals are dropped.
    """
    # Group detections to intervals
    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]

    # Remove empty intervals [x, x]
    return [(g[0], g[-1]) for g in groups if g[0] != g[-1]]


def _find_horizontal_lines(img_gray: np.ndarray, config: SegmentationCoreTrimConfig) -> list[int]:
    """Detect y-coordinates of prominent horizontal lines (e.g. tray dividers) in the image.

    Runs a Hough transform on the horizontal-edge map (Prewitt), restricted to exactly
    horizontal lines (a single fixed angle), keeping only the strongest peaks. Used to split
    the tray into bands for top/bottom trimming.

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
        [int(dist) for _, _, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance), strict=True)]
    )
    # Remove line too close from border
    y_lines = [y_line for y_line in y_lines if y_line - min_distance > 0 and y_line + min_distance < img_gray.shape[0]]
    return [0] + y_lines + [img_gray.shape[0]]


def _find_left_right_foreground_intervals(
    img_hsv: np.ndarray, config: SegmentationCoreTrimConfig
) -> list[tuple[int, int]]:
    """Find the column intervals to keep after trimming black background on the left/right.

    Falls back to the full image width if no interval survives the minimum width filter.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Surviving column intervals based on black-background
            trimming (falls back to the full image width if empty).
    """
    # Get all segments that are valid and drop short ones
    feature_col_foreground = _in_range(img_hsv[:, :, 2], threshold_high=config.background_val_threshold).mean(axis=0)
    indicator_col_foreground = np.nonzero(feature_col_foreground < config.background_val_vratio)[0]
    min_segment_len = config.downscale_factor * config.min_segment_width_px

    lr_trims_foreground = [
        lr_trim
        for lr_trim in _find_valid_intervals(detections=indicator_col_foreground)
        if min_segment_len <= lr_trim[1] - lr_trim[0]
    ]

    if len(lr_trims_foreground) == 0:
        lr_trims_foreground = [(0, img_hsv.shape[1] - 1)]

    return lr_trims_foreground


def _find_left_right_wood_intervals(img_hsv: np.ndarray, config: SegmentationCoreTrimConfig) -> list[tuple[int, int]]:
    """Find the column intervals that contain wooden blocks.

    This is useful for identifier wooden trays, as well as "Kernverlust" (loss of core)
    placeholder blocks.

    Args:
        img_hsv (np.ndarray): HSV image of the cropped tray region.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Column intervals based on wood-color (hue/saturation).
    """
    # Get all segments that are valid and drop short ones
    min_segment_len = config.downscale_factor * config.min_segment_width_px

    feature_col_wood = (
        _in_range(img_hsv[:, :, 0], config.wood_min_hue, config.wood_max_hue)
        & _in_range(img_hsv[:, :, 1], config.wood_min_sat, config.wood_max_sat)
        & _in_range(img_hsv[:, :, 2], threshold_low=config.wood_min_value)
    ).mean(axis=0)
    indicator_col_wood = np.nonzero(feature_col_wood >= config.wood_vratio)[0]

    return [
        lr_trim
        for lr_trim in _find_valid_intervals(detections=indicator_col_wood)
        if min_segment_len <= lr_trim[1] - lr_trim[0]
    ]


def _find_top_bottom_intervals(
    img_hsv: np.ndarray,
    y_lines: list[int],
    wood_lengths: list[int],
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
        y_lines (list[int]): Row indices of detected horizontal dividers, sorted ascending.
        wood_lengths (list[int]): Total length of the identified wood segments per row.
        config (SegmentationCoreTrimConfig): Tunable segmentation parameters.

    Returns:
        list[tuple[int, int]]: Start and end row indices of each candidate top/bottom
            interval, sorted by score in descending order.
    """
    if not wood_lengths:
        return []
    max_wood_length = max(wood_lengths)

    height, width, _ = img_hsv.shape
    intervals = []
    current_interval = []

    for start, end, wood_length in zip(y_lines[:-1], y_lines[1:], wood_lengths, strict=True):
        if (
            wood_length > config.max_wood_per_row_relative_to_max_wood * max_wood_length
            and wood_length > config.max_wood_per_row_relative_to_width * width
        ):
            # too much wood, ignore this segment
            is_valid = False
        else:
            im_segment = img_hsv[start:end, :, :]

            # Slice out 10% from top, bottom, left, and right
            segment_height, segment_width = im_segment.shape[:2]
            margin_height = int(segment_height * config.background_margin_percentage)
            margin_width = int(segment_width * config.background_margin_percentage)
            cropped_img = im_segment[
                margin_height : segment_height - margin_height, margin_width : segment_width - margin_width
            ]

            im_hue, im_sat, im_val = np.moveaxis(cropped_img, 2, 0)

            # Background must be both low on texture (based on Scharr oparator of the hue and saturation channels) as
            # well as dark (based on value channel)
            hue_scharr = scharr(im_hue)
            sat_scharr = scharr(im_sat)
            background_score = np.sqrt(sat_scharr**2 + hue_scharr**2) * im_val
            is_valid = np.median(background_score) > config.max_background_score

        if is_valid:
            current_interval.append((start, end, wood_length))
        elif current_interval:
            intervals.append(current_interval)
            current_interval = []

    if current_interval:
        intervals.append(current_interval)

    detections = []
    for interval in intervals:
        # remove segments at the top/bottom of the interval that don't haven sufficient non-wood
        max_non_wood_length = max(width - wood_length for _, _, wood_length in interval)
        valid_segments = [
            (start, end)
            for start, end, wood_length in interval
            if width - wood_length > config.min_non_wood_per_row_relative_to_max * max_non_wood_length
        ]
        # there will always be at least one valid segment
        detections.append((valid_segments[0][0], valid_segments[-1][1] - 1))

    return (
        # Look for interval that is large (x1-x0) and close to center (1 - |(x1+x0-H)/H|)
        sorted(
            detections,
            key=lambda x: (x[1] - x[0]) * (1 - abs((x[1] + x[0] - height) / height)),
            reverse=True,
        )
        if detections
        else [(0, height - 1)]
    )


def segment_core(
    img_metadata: ImageMetadataCores, tray: ImageSegmentResult, config: SegmentationCoreTrimConfig
) -> CoreSegmentResult:
    """Trim the bounding box to exclude the wooden tray and black background around the core.

    Trims left/right based on the value (brightness) channel to drop black background for the
    outer bbox; the surviving sub-segments used for `bbox_segments` are additionally restricted
    to columns that aren't wood-colored (hue/saturation). Trims top/bottom by splitting the tray
    into row bands at Hough-detected horizontal divider lines, then keeping the bands that don't
    match a wooden-tray profile (saturation, hue, and edge texture) and aren't black background.

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
    lr_trims_foreground = _find_left_right_foreground_intervals(img_hsv=hsv, config=config)
    left_trim_background = lr_trims_foreground[0][0]
    right_trim_background = lr_trims_foreground[-1][1]

    # Remove wood / background (top/bottom). Pick the best-scored interval (large and
    # close to vertical center)
    y_lines = _find_horizontal_lines(img_gray=gray, config=config)

    trimmed_hsv = hsv[:, left_trim_background:right_trim_background, :]

    wood_lengths = []
    wood_bboxes = []

    for start, end in zip(y_lines[:-1], y_lines[1:], strict=True):
        wood_intervals = _find_left_right_wood_intervals(trimmed_hsv[start:end, :, :], config)
        wood_bboxes.extend([(x0, start, x1, end) for x0, x1 in wood_intervals])
        wood_lengths.append(sum([wood_end - wood_start for wood_start, wood_end in wood_intervals]))

    top_trim, bottom_trim = _find_top_bottom_intervals(
        img_hsv=trimmed_hsv,
        y_lines=y_lines,
        wood_lengths=wood_lengths,
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
        wood_bboxes=[
            scale_bbox(
                bbox=(
                    x_min + left_trim_background + x0,
                    y_min + y0,
                    x_min + left_trim_background + x1,
                    y_min + y1,
                ),
                factor=1 / config.downscale_factor,
            )
            for x0, y0, x1, y1 in wood_bboxes
        ],
        time=timer() - t_start,
        y_lines=((y_min + np.array(y_lines)) / config.downscale_factor).tolist(),
    )
