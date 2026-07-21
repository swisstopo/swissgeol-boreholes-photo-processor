"""Module for checking core width and length consistency across a set of detections."""

from collections.abc import Callable
from typing import TypeVar

import numpy as np

from src.evaluations.config import (
    CoreCheckConfig,
    CoreCheckResult,
    CoreLengthCheckConfig,
    CoreLengthCheckResult,
    CoreWidthCheckConfig,
    CoreWidthCheckResult,
    EvaluationConfig,
)
from src.models import ImageMetadataProcessed

_CoreCheckResultT = TypeVar("_CoreCheckResultT", CoreWidthCheckResult, CoreLengthCheckResult)


def _check_core(
    config: CoreCheckConfig,
    values: list[float],
    scales: list[float],
) -> list[tuple[float, float, float, bool] | None]:
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
        list[tuple[float, float, float, bool] | None]: One entry per detection, aligned
        1:1 with values/scales. An entry is None if there aren't enough detections for a reliable
        reference, or if this core's expected value is 0 (undefined relative error). Otherwise it's
        a tuple of (value, expected, relative_error, passed).
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
        relative_error = abs(value - expected) / expected
        passed = bool(relative_error <= config.relative_tolerance)
        results.append((value, expected, relative_error, passed))

    return results


def _build_results(
    raw_results: list[tuple[float, float, float, bool] | None],
    build: Callable[[float, float, float, bool], _CoreCheckResultT],
) -> list[_CoreCheckResultT | None]:
    """Apply build to each non-None _check_core entry, passing None through unchanged."""
    return [build(*raw_result) if raw_result is not None else None for raw_result in raw_results]


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

    return _build_results(
        _check_core(config, widths, scales),
        lambda value, expected, relative_error, passed: CoreWidthCheckResult(
            passed=passed, measure_px=value, reference_px=expected, relative_error=relative_error
        ),
    )


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

    return _build_results(
        _check_core(config, lengths, scales),
        lambda value, expected, relative_error, passed: CoreLengthCheckResult(
            passed=passed,
            measure_px=value,
            reference_px=expected,
            relative_error=relative_error,
        ),
    )


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
