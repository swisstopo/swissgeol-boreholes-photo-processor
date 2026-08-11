"""Shared package utilities."""

from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.transform import rescale

from src.config import SegmentationError

# Cores are always TIFF; cuttings additionally allow common photo formats, which
# tifffile cannot decode, so those are routed through PIL instead.
_TIFF_EXTENSIONS = {".tif", ".tiff"}


# Store up to 4 different image (downscaled) in memory
@lru_cache(maxsize=4)
def load_image(path: str, factor: float = 1.0) -> np.ndarray:
    """Load an image and normalize it to an RGB float array in [0, 1].

    Only 3-channel images are supported; grayscale (2D) input raises an error.
    TIFs are read via tifffile since raw borehole scans may be 16-bit, which
    PIL does not handle as reliably for downstream processing; other formats
    (e.g. JPEG, BMP) are read via PIL.

    Args:
        path (str): Path to the image to load.
        factor (float): Downscale factor applied after loading; 1.0 leaves the image unscaled.

    Returns:
        np.ndarray: RGB image array with float values in [0, 1].

    Raises:
        SegmentationError: If the image is not a 3/4-channel array.
        ValueError: If the image has an unsupported dtype.
    """
    img = tifffile.imread(path) if Path(path).suffix.lower() in _TIFF_EXTENSIONS else np.array(Image.open(path))

    if img.ndim != 3:
        raise SegmentationError(f"Input should be RGB(A): {path}")

    # drop alpha channel, e.g. from RGBA scans
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] != 3:
        raise SegmentationError(f"Input should be RGB(A): {path}")

    # normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(float) / 65535.0
    else:
        raise ValueError(f"Unsupported image data type ({img.dtype}): {path}")

    return rescale(img, factor, channel_axis=-1, anti_aliasing=True) if factor != 1.0 else img


@cache
def get_image_shape(path: str) -> tuple[int, int, int]:
    """Get the shape of an image without loading the entire image into memory.

    Args:
        path (str): Path to the image.

    Returns:
        tuple[int, int, int]: The shape of the image as (height, width, channels).

    Raises:
        SegmentationError: If the image is not RGB/RGBA.
    """
    if Path(path).suffix.lower() in _TIFF_EXTENSIONS:
        with tifffile.TiffFile(path) as tif:
            img_shape = tif.pages[0].shape
    else:
        with Image.open(path) as img:
            img_shape = (img.height, img.width, len(img.getbands()))

    if len(img_shape) != 3 or img_shape[-1] not in (3, 4):
        raise SegmentationError(f"Input should be RGB: {path}")
    # RGBA scans get their alpha channel dropped by load_image, so report the RGB shape
    return (*img_shape[:2], 3)


def scale_bbox(
    bbox: tuple[float, float, float, float],
    factor: float = 1.0,
) -> tuple[float, float, float, float]:
    """Scale a bounding box's coordinates by a constant factor.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box.
        factor (float, optional): Multiplicative scale factor. Defaults to 1.0.

    Returns:
        tuple[float, float, float, float]: The scaled bounding box coordinates.
    """
    scaled = np.array(bbox) * factor
    return (scaled[0].item(), scaled[1].item(), scaled[2].item(), scaled[3].item())
