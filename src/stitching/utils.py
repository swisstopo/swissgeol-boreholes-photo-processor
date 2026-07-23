"""Helper functions for stitching."""

import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _resize_images(
    images: list[Image.Image],
    scales: list[float],
    max_core_height: int,
    max_core_width: int,
) -> list[Image.Image]:
    """Resize each core crop by its own scale factor, clamped to fit within the max dimensions.

    Args:
        images (list[Image.Image]): Raw core crops to resize.
        scales (list[float]): Per-image resize factor (pixels-per-unit ratio), same order as images.
        max_core_height (int): Maximum allowed height in pixels after resizing.
        max_core_width (int): Maximum allowed width in pixels after resizing.

    Returns:
        list[Image.Image]: Cores resized to a consistent pixel scale.
    """
    cores_resized: list[Image.Image] = []
    for img, scale in zip(images, scales, strict=True):
        # Ensure resize falls within range
        if img.width * scale > max_core_width or img.height * scale > max_core_height:
            logger.warning(f"Image {img.size} ({scale=:.4f}) cannot be fit in ({max_core_width}, {max_core_height})")
            scale = min(max_core_width / img.width, max_core_height / img.height)

        cores_resized.append(
            img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        )

    return cores_resized
