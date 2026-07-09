"""Module for checking core length consistency across a set of detections."""

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from src.config import CoreLengthCheckConfig, CoreLengthCheckResults
from src.models import ImageMetadataProcessed


def check_core_length(
    detections: list[ImageMetadataProcessed], config: CoreLengthCheckConfig
) -> list[CoreLengthCheckResults]:
    """Flag cores whose length deviates too far from the expected length based on the folder's RANSAC fit.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (CoreLengthCheckConfig): Configuration parameters for the core length check.

    Returns:
        list[CoreLengthCheckResults]: List of results for each core.
        If not enough detections are present, returns an empty list.
    """
    # check min number of detections
    if len(detections) < config.min_samples:
        return []

    # extract lengths and expected lengths per core
    lengths = [d.result.bounding_box[2] - d.result.bounding_box[0] for d in detections]

    # robustly fit a single px-per-metre ratio for the whole folder (line through the origin)
    depths_m = np.array([d.depth_end - d.depth_start for d in detections]).reshape(-1, 1)
    ransac = RANSACRegressor(LinearRegression(fit_intercept=False))
    ransac.fit(depths_m, lengths)
    folder_ratio_px_per_m = ransac.estimator_.coef_[0]

    expected_lengths = [(d.depth_end - d.depth_start) * folder_ratio_px_per_m for d in detections]

    # check deviation for each core
    results = []
    for length, expected_length, detection in zip(lengths, expected_lengths, detections, strict=True):
        if expected_length == 0:
            continue
        deviation = abs(length - expected_length) / expected_length
        passed = bool(deviation <= config.relative_tolerance)
        res = CoreLengthCheckResults(
            filename=detection.image_path.name,
            length_px=length,
            expected_length_px=expected_length,
            folder_ratio_px_per_m=folder_ratio_px_per_m,
            deviation=deviation,
            passed=passed,
        )
        results.append(res)

    return results
