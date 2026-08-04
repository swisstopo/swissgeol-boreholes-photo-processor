"""Core bbox detection: trimming the wooden tray and black background around the core."""

from itertools import groupby
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2hsv

from src.config import SegmentationCoreConfig, SegmentationError
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
        raise SegmentationError("Entire region classified as invalid; no valid interval found")

    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]
    results = np.array([[g[0], g[-1]] for g in groups])

    return [tuple(x) for x in sorted(results.tolist(), key=lambda x: x[1] - x[0], reverse=True)]


def segment_core(
    img_metadata: ImageMetadata, tray: ImageSegmentResult, config: SegmentationCoreConfig
) -> CoreSegmentResult:
    """Trim the bounding box to exclude the wooden tray and black background around the core.

    Trims left/right based on the value (brightness) channel to drop black background, and
    trims top/bottom based on both the saturation channel (wooden tray) and the value channel
    (black background), keeping the intersection of the two vertical trims.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and trim.
        tray (ImageSegmentResult): Bounding box to trim, in the original image's coordinate space.
        config (SegmentationCoreConfig): Tunable segmentation parameters.

    Returns:
        CoreSegmentResult: bbox is the trimmed bounding box as (left, top, right, bottom), in the
            original image's coordinate space. bbox_segments holds one bbox per surviving
            left/right sub-segment.

    Raises:
        SegmentationError: If every row or column is classified as tray/background (no valid
            interval found) for any of the three trim passes.
    """
    t_start = timer()
    img = img_metadata.load_image(factor=config.downscale_factor)
    x_min, y_min, x_max, y_max = scale_bbox(tray.bbox, factor=config.downscale_factor)

    hsv = rgb2hsv(img[round(y_min) : round(y_max + 1), round(x_min) : round(x_max + 1)])

    # Remove black background (left/right)
    # Get all segments that are valid and drop short ones
    lr_trims = _find_valid_intervals(
        values=-hsv[:, :, 2].T,
        threshold=-config.background_val_threshold,
        ratio=config.background_val_vratio,
    )
    lr_trims = [
        lr_trim
        for lr_trim in lr_trims
        if config.downscale_factor * config.min_segment_height_px <= lr_trim[1] - lr_trim[0]
    ]

    if len(lr_trims) == 0:
        lr_trims = [(0, hsv.shape[1] - 1)]

    left_trim_background = np.array(lr_trims)[:, 0].min().item()
    right_trim_background = np.array(lr_trims)[:, 1].max().item()

    # Remove wood / background (top/bottom). If multiple intervals, consider only largest one (first)
    top_trim_wood, bottom_trim_wood = _find_valid_intervals(
        values=np.concatenate([hsv[:, lr[0] : lr[1] + 1, 1] for lr in lr_trims], axis=1),
        threshold=config.wood_sat_threshold,
        ratio=config.wood_sat_hratio,
    )[0]

    top_trim_background, bottom_trim_background = _find_valid_intervals(
        values=-np.concatenate([hsv[:, lr[0] : lr[1] + 1, 2] for lr in lr_trims], axis=1),
        threshold=-config.background_val_threshold,
        ratio=config.background_val_hratio,
    )[0]

    return CoreSegmentResult(
        bbox=scale_bbox(
            bbox=(
                x_min + left_trim_background,
                y_min + max(top_trim_background, top_trim_wood),
                x_min + right_trim_background,
                y_min + min(bottom_trim_background, bottom_trim_wood),
            ),
            factor=1 / config.downscale_factor,
        ),
        bbox_segments=[
            scale_bbox(
                bbox=(
                    x_min + left_trim,
                    y_min + max(top_trim_background, top_trim_wood),
                    x_min + right_trim,
                    y_min + min(bottom_trim_background, bottom_trim_wood),
                ),
                factor=1 / config.downscale_factor,
            )
            for left_trim, right_trim in lr_trims
        ],
        time=timer() - t_start,
    )
