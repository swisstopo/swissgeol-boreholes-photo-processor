"""Module for checking core width and length consistency across a set of detections."""

import numpy as np

from src.config import (
    CoreCheckConfig,
    CoreCheckResult,
    CoreLengthCheckConfig,
    CoreLengthCheckResult,
    CoreWidthCheckConfig,
    CoreWidthCheckResult,
    EvaluationConfig,
)
from src.models import ImageMetadataProcessed


def _check_core(
    config: CoreCheckConfig,
    values: list[float],
    scales: list[float],
) -> list[tuple[float, float, float, float, bool] | None]:
    """Flag cores whose measured value deviates too far from a robust folder-wide reference.

    The reference is the median of each core's own value/scale ratio, which is insensitive
    to outlier cores. A core's expected value is then scale * reference, so passing scale=1
    for every core (the width check) makes the reference a plain median of the raw values,
    while passing the depth extent (the length check) turns it into a px-per-metre ratio.

    Args:
        config (CoreCheckConfig): Common tunable parameters (relative_tolerance, min_samples).
        values (list[float]): Measured value (width or length, in pixels) for each detection.
        scales (list[float]): Per-core scale (1.0 for width, depth extent for length).

    Returns:
        list[tuple[float, float, float, float, bool] | None]: One entry per detection, aligned
        1:1 with values/scales. An entry is None if there aren't enough detections for a reliable
        reference, or if this core's expected value is 0 (undefined deviation). Otherwise it's a
        tuple of (value, expected, reference, deviation, passed).
    """
    if len(values) < config.min_samples:
        return [None] * len(values)

    values_arr = np.array(values)
    scales_arr = np.array(scales)
    ratios = values_arr[scales_arr != 0] / scales_arr[scales_arr != 0]
    folder_reference = float(np.median(ratios))

    results = []
    for value, scale in zip(values, scales, strict=True):
        expected = scale * folder_reference
        if expected == 0:
            results.append(None)
            continue
        deviation = abs(value - expected) / expected
        passed = bool(deviation <= config.relative_tolerance)
        results.append((value, expected, folder_reference, deviation, passed))

    return results


def check_core_width(
    detections: list[ImageMetadataProcessed], config: CoreWidthCheckConfig
) -> list[CoreWidthCheckResult | None]:
    """Flag cores whose width deviates too far from the folder's median width.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreWidthCheckConfig): Configuration parameters for the core width check.

    Returns:
        list[CoreWidthCheckResult | None]: One entry per detection, aligned 1:1 with detections.
        None where the check was skipped for that core (see CoreCheckConfig.min_samples).
    """
    widths = [d.result.bounding_box[3] - d.result.bounding_box[1] for d in detections]
    scales = [1.0] * len(detections)

    results = []
    for result in _check_core(config, widths, scales):
        if result is None:
            results.append(None)
            continue
        value, _, reference, deviation, passed = result
        results.append(
            CoreWidthCheckResult(passed=passed, width=value, folder_median_width=reference, deviation=deviation)
        )
    return results


def check_core_length(
    detections: list[ImageMetadataProcessed], config: CoreLengthCheckConfig
) -> list[CoreLengthCheckResult | None]:
    """Flag cores whose length deviates too far from the expected length based on the folder's median ratio.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreLengthCheckConfig): Configuration parameters for the core length check.

    Returns:
        list[CoreLengthCheckResult | None]: One entry per detection, aligned 1:1 with detections.
        None where the check was skipped for that core (see CoreCheckConfig.min_samples).
    """
    lengths = [d.result.bounding_box[2] - d.result.bounding_box[0] for d in detections]
    scales = [d.depth_end - d.depth_start for d in detections]

    results = []
    for result in _check_core(config, lengths, scales):
        if result is None:
            results.append(None)
            continue
        value, expected, reference, deviation, passed = result
        results.append(
            CoreLengthCheckResult(
                passed=passed,
                length_px=value,
                expected_length_px=expected,
                folder_ratio_px_per_m=reference,
                deviation=deviation,
            )
        )
    return results


def check_core(detections: list[ImageMetadataProcessed], config: EvaluationConfig) -> list[CoreCheckResult]:
    """Run the width and length checks and merge them into one result per file.

    width/length checks report results in different units and reference a differently-scaled
    folder reference, so they can't be merged into a single flat result type. Instead, each
    file gets one CoreCheckResult with both nested inside, keyed by filename.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (EvaluationConfig): Configuration parameters for all core checks.

    Returns:
        list[CoreCheckResult]: One result per detection. width/length are None for a file
        whose corresponding check was skipped (see CoreCheckConfig.min_samples).
    """
    width_results = check_core_width(detections, config.core_width)
    length_results = check_core_length(detections, config.core_length)

    return [
        CoreCheckResult(filename=detection.image_path.name, width=width, length=length)
        for detection, width, length in zip(detections, width_results, length_results, strict=True)
    ]
