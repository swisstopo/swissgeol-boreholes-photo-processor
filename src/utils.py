"""Shared packages utilities."""

from functools import cache

import numpy as np
import tifffile
from skimage.transform import rescale


def load_image(path: str, factor: float = 1.0) -> np.ndarray:
    """Load a TIF image and normalize it to an RGB float array in [0, 1].

    Only RGB (3-channel) images are supported; grayscale (2D) input raises an error.
    Uses tifffile instead of PIL since raw borehole scans may be 16-bit, which
    PIL does not handle as reliably for downstream processing.

    Args:
        path (str): Path to the TIF image to load.
        factor (float): Downscale factor applied after loading; 1.0 leaves the image unscaled.

    Returns:
        np.ndarray: RGB image array with float values in [0, 1].
    """
    img = tifffile.imread(path)

    if img.ndim == 2:
        raise ValueError(f"Input should be RGB: {path}")

    # drop alpha channel, e.g. from RGBA scans
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] != 3:
        raise ValueError(f"Input should be RGB: {path}")

    # normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(float) / 65535.0
    else:
        raise ValueError(f"Unsupported TIF data type ({img.dtype}): {path}")

    return rescale(img, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img


@cache
def get_image_shape(path: str) -> tuple[int, int, int]:
    """Get the shape of a TIF image without loading the entire image into memory.

    Args:
        path (str): Path to the TIF image.

    Returns:
        tuple[int, int, int]: The shape of the image as (height, width, channels).
    """
    with tifffile.TiffFile(path) as tif:
        img_shape = tif.pages[0].shape
        if len(img_shape) != 3 or img_shape[-1] not in (3, 4):
            raise ValueError(f"Input should be RGB: {path}")
        # RGBA scans get their alpha channel dropped by load_image, so report the RGB shape
        return (*img_shape[:2], 3)


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
