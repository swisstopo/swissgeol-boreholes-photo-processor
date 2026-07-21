"""Shared packages utilities."""

from functools import lru_cache

import numpy as np
import tifffile
from skimage.transform import rescale


# Store up to 4 different image (downscaled) in memory
@lru_cache(maxsize=4)
def load_image(path: str, factor: float = 1.0) -> np.ndarray:
    """Load a TIF image and normalize it to an RGB float array in [0, 1].

    Only 3-channel TIFs are supported; grayscale (2D) input raises an error.
    Uses tifffile instead of PIL since raw borehole scans may be 16-bit, which
    PIL does not handle as reliably for downstream processing.

    Args:
        path (str): Path to the TIF image to load.
        factor (float): Downscale factor applied after loading; 1.0 leaves the image unscaled.

    Returns:
        np.ndarray: RGB image array with float values in [0, 1].
    """
    img = tifffile.imread(path)

    # grayscale to RGB
    if img.ndim == 2:
        raise ValueError(f"Input should be RGB: {path}")

    # normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(float) / 65535.0
    else:
        raise ValueError(f"Unsupported TIF data type ({img.dtype}): {path}")

    return rescale(img, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img


def scale_bbox(
    bbox: tuple[float, float, float, float],
    factor: float = 1,
) -> tuple[float, float, float, float]:
    """Scale a bounding box's coordinates by a constant factor.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box.
        factor (float, optional): Multiplicative scale factor. Defaults to 1.

    Returns:
        tuple[float, float, float, float]: The scaled bounding box coordinates.
    """
    scaled = np.array(bbox) * factor
    return (scaled[0].item(), scaled[1].item(), scaled[2].item(), scaled[3].item())
