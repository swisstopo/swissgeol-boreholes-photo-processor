"""Fix common mistakes where the detected core is too narrow."""

import logging

import numpy as np
from skimage.color import rgb2hsv

from src.config import SegmentationConfig
from src.models import ImageMetadataProcessedCores

logger = logging.getLogger(__name__)


def _check_wood_ratio(img: np.ndarray, bbox: tuple[float, float, float, float], config: SegmentationConfig) -> float:
    """Check the ratio of wood pixels in a cropped region.

    Args:
        img (np.ndarray): RGB image array.
        bbox (tuple[float, float, float, float]): Bounding box (left, top, right, bottom).
        config (SegmentationConfig): Segmentation configuration.

    Returns:
        float: Ratio of wood pixels in the cropped region.
    """
    left, top, right, bottom = (int(v) for v in bbox)
    cropped = img[top:bottom, left:right]

    if cropped.size == 0:
        return 0.0

    # Convert to HSV for wood detection
    cropped_hsv = rgb2hsv(cropped)

    # Detect wood using the same parameters as core segmentation
    core_config = config.core.core
    feature_wood = (
        (core_config.wood_min_hue < cropped_hsv[:, :, 0])
        & (cropped_hsv[:, :, 0] < core_config.wood_max_hue)
        & (core_config.wood_min_sat < cropped_hsv[:, :, 1])
        & (cropped_hsv[:, :, 1] < core_config.wood_max_sat)
    )

    return feature_wood.mean()


def _find_closest_y_line(
    y_lines: list[float] | None,
    target: float,
    max_diff: float,
) -> float | None:
    """Find the y_line closest to the target value within the maximum allowed difference.

    Args:
        y_lines (list[float] | None): List of y_line coordinates, or None.
        target (float): Target coordinate to match.
        max_diff (float): Maximum allowed difference between y_line and target.

    Returns:
        float | None: The closest y_line within max_diff, or None if no suitable line found.
    """
    if y_lines is None or len(y_lines) == 0:
        return None

    # Find the closest y_line to the target
    closest_line = min(y_lines, key=lambda y: abs(y - target))
    diff = abs(closest_line - target)

    # Only return if within the allowed difference
    return closest_line if diff <= max_diff else None


def fix_narrow_cores(
    detections: list[ImageMetadataProcessedCores],
    config: SegmentationConfig,
) -> list[ImageMetadataProcessedCores]:
    """Fix common mistakes where the detected core is too narrow.

    Iterates through detections sorted by depth and applies a fallback when:
    - Current and previous input images have identical dimensions
    - There's >10% difference in top/bottom coordinates relative to previous core height
    - Applying previous bbox to current image has >=25% wood content

    When fixing, uses the closest y_lines from the current image if available,
    otherwise falls back to the previous image's top/bottom values directly.

    Args:
        detections (list[ImageMetadataProcessedCores]): List of processed image detections.
        config (SegmentationConfig): Segmentation configuration.

    Returns:
        list[ImageMetadataProcessedCores]: Updated detections with fixed narrow cores.
    """
    if len(detections) < 2:
        return detections

    # Sort by depth to ensure sequential processing
    sorted_detections = sorted(detections, key=lambda d: d.depth_start)

    for i in range(1, len(sorted_detections)):
        prev = sorted_detections[i - 1]
        curr = sorted_detections[i]

        # Skip if either detection has no core
        if prev.core is None or curr.core is None:
            continue

        # Condition 1: Check if image dimensions are identical
        if prev.shape[:2] != curr.shape[:2]:
            continue

        # Get bounding boxes
        prev_bbox = prev.core.bbox
        curr_bbox = curr.core.bbox

        # Calculate previous core height
        prev_height = prev_bbox[3] - prev_bbox[1]  # bottom - top
        if prev_height == 0:
            continue

        # Condition 2: Check for >10% difference in top or bottom coordinates
        top_diff = abs(curr_bbox[1] - prev_bbox[1])
        bottom_diff = abs(curr_bbox[3] - prev_bbox[3])

        # Calculate the maximum allowed difference (10% of previous height)
        max_diff = 0.10 * prev_height

        if top_diff <= max_diff and bottom_diff <= max_diff:
            continue

        # Condition 3: Check wood ratio when applying previous bbox to current image
        curr_img = curr.load_image()
        wood_ratio = _check_wood_ratio(curr_img, prev_bbox, config)

        if wood_ratio >= 0.25:
            # All conditions met - override current bbox with previous

            # Try to find closest y_lines for top and bottom
            new_top = _find_closest_y_line(curr.core.y_lines, prev_bbox[1], max_diff)
            new_bottom = _find_closest_y_line(curr.core.y_lines, prev_bbox[3], max_diff)

            # Fall back to previous values if no suitable y_lines found
            if new_top is None:
                new_top = prev_bbox[1]
            if new_bottom is None:
                new_bottom = prev_bbox[3]

            logger.info(f"Applying fallback for narrow core for {curr.image_path.name}.")
            curr.core.bbox = (curr_bbox[0], new_top, curr_bbox[2], new_bottom)

    return sorted_detections
