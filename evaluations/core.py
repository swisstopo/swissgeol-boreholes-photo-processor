"""Module for checking core width and length consistency across a set of detections."""

import numpy as np

from src.config import (
    CoreCheckConfig,
    CoreLengthCheckConfig,
    CoreLengthCheckResults,
    CoreWidthCheckConfig,
    CoreWidthCheckResults,
)
from src.models import ImageMetadataProcessed


def _check_core(
    detections: list[ImageMetadataProcessed],
    config: CoreCheckConfig,
    values: list[float],
    scales: list[float],
) -> list[tuple[ImageMetadataProcessed, float, float, float, float, bool]]:
    """Flag cores whose measured value deviates too far from a robust folder-wide reference.

    The reference is the median of each core's own value/scale ratio, which is insensitive
    to outlier cores. A core's expected value is then scale * reference, so passing scale=1
    for every core (the width check) makes the reference a plain median of the raw values,
    while passing the depth extent (the length check) turns it into a px-per-metre ratio.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreCheckConfig): Common tunable parameters (relative_tolerance, min_samples).
        values (list[float]): Measured value (width or length, in pixels) for each detection.
        scales (list[float]): Per-core scale (1.0 for width, depth extent for length).

    Returns:
        list[tuple[ImageMetadataProcessed, float, float, float, float, bool]]: For each detection with a
        non-zero expected value, a tuple of (detection, value, expected, reference, deviation, passed).
        If not enough detections are present, returns an empty list.
    """
    if len(detections) < config.min_samples:
        return []

    values_arr = np.array(values)
    scales_arr = np.array(scales)
    ratios = values_arr[scales_arr != 0] / scales_arr[scales_arr != 0]
    folder_reference = float(np.median(ratios))

    results = []
    for detection, value, scale in zip(detections, values, scales, strict=True):
        expected = scale * folder_reference
        if expected == 0:
            continue
        deviation = abs(value - expected) / expected
        passed = bool(deviation <= config.relative_tolerance)
        results.append((detection, value, expected, folder_reference, deviation, passed))

    return results


def check_core_width(
    detections: list[ImageMetadataProcessed], config: CoreWidthCheckConfig
) -> list[CoreWidthCheckResults]:
    """Flag cores whose width deviates too far from the folder's median width.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreWidthCheckConfig): Configuration parameters for the core width check.

    Returns:
        list[CoreWidthCheckResults]: List of results for each core.
        If not enough detections are present, returns an empty list.
    """
    widths = [d.result.bounding_box[3] - d.result.bounding_box[1] for d in detections]
    scales = [1.0] * len(detections)

    return [
        CoreWidthCheckResults(
            filename=detection.image_path.name,
            width=value,
            folder_median_width=reference,
            deviation=deviation,
            passed=passed,
        )
        for detection, value, _, reference, deviation, passed in _check_core(detections, config, widths, scales)
    ]


def check_core_length(
    detections: list[ImageMetadataProcessed], config: CoreLengthCheckConfig
) -> list[CoreLengthCheckResults]:
    """Flag cores whose length deviates too far from the expected length based on the folder's median ratio.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreLengthCheckConfig): Configuration parameters for the core length check.

    Returns:
        list[CoreLengthCheckResults]: List of results for each core.
        If not enough detections are present, returns an empty list.
    """
    lengths = [d.result.bounding_box[2] - d.result.bounding_box[0] for d in detections]
    scales = [d.depth_end - d.depth_start for d in detections]

    return [
        CoreLengthCheckResults(
            filename=detection.image_path.name,
            length_px=value,
            expected_length_px=expected,
            folder_ratio_px_per_m=reference,
            deviation=deviation,
            passed=passed,
        )
        for detection, value, expected, reference, deviation, passed in _check_core(
            detections, config, lengths, scales
        )
    ]
