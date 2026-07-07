"""Module for checking core width consistency across a set of detections."""

import numpy as np

from src.config import CoreWidthCheckConfig, CoreWidthCheckResults
from src.models import ImageMetadataProcessed


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
    # check min number of detections
    if len(detections) < config.min_samples:
        return []

    # extract width per core
    widths = [d.result.bounding_box[2] - d.result.bounding_box[0] for d in detections]

    # folder median
    folder_median = np.median(widths)

    # check deviation for each core
    results = []
    for width, detection in zip(widths, detections, strict=True):
        deviation = abs(width - folder_median) / folder_median
        if deviation > config.relative_tolerance:
            res = CoreWidthCheckResults(
                filename=detection.image_path.name,
                width=width,
                folder_median_width=folder_median,
                deviation=deviation,
                passed=False,
            )
            results.append(res)
        else:
            res = CoreWidthCheckResults(
                filename=detection.image_path.name,
                width=width,
                folder_median_width=folder_median,
                deviation=deviation,
                passed=True,
            )
            results.append(res)

    return results
