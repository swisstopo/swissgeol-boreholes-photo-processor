"""Helper functions for stitching."""

import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _resize_scale(width: int, height: int, scale: float, max_width: int, max_height: int) -> float:
    """Clamp a proposed resize scale so neither resulting dimension exceeds its maximum.

    Args:
        width (int): Unscaled width in pixels.
        height (int): Unscaled height in pixels.
        scale (float): Proposed resize factor.
        max_width (int): Maximum allowed width in pixels after resizing.
        max_height (int): Maximum allowed height in pixels after resizing.

    Returns:
        float: scale, or the largest scale that keeps both dimensions within range if scale overflows either.
    """
    if width * scale > max_width or height * scale > max_height:
        return min(max_width / width, max_height / height)
    return scale


def _resize_images(
    images: list[Image.Image],
    scales: list[float],
    max_core_height: int,
    max_core_width: int,
) -> tuple[list[Image.Image], list[bool]]:
    """Resize each core crop by its own scale factor, clamped to fit within the max dimensions.

    Args:
        images (list[Image.Image]): Raw core crops to resize.
        scales (list[float]): Per-image resize factor (pixels-per-unit ratio), same order as images.
        max_core_height (int): Maximum allowed height in pixels after resizing.
        max_core_width (int): Maximum allowed width in pixels after resizing.

    Returns:
        tuple[list[Image.Image], list[bool]]: Cores resized to a consistent pixel scale and indicator to signify if
            core could be rescaled in the defined ranges.
    """
    cores_resized: list[Image.Image] = []
    valid_resizes: list[bool] = []

    for img, scale in zip(images, scales, strict=True):
        clamped_scale = _resize_scale(img.width, img.height, scale, max_core_width, max_core_height)
        if clamped_scale != scale:
            logger.warning(f"Image {img.size} ({scale=:.4f}) cannot be fit in ({max_core_width}, {max_core_height})")
        valid_resizes.append(clamped_scale == scale)

        cores_resized.append(
            img.resize(
                (max(round(img.width * clamped_scale), 1), max(round(img.height * clamped_scale), 1)),
                Image.Resampling.LANCZOS,
            )
        )

    return cores_resized, valid_resizes
