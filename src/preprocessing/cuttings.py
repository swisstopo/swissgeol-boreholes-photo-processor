"""Module for preprocessing the cuttings images."""

import glob
import logging
from pathlib import Path

import tifffile

from src.models import ImageMetadataCuttings

# TODO: document in readme which file extensions we support
_CUTTINGS_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_cuttings(input_dir: Path) -> list[ImageMetadataCuttings]:
    """Collect cuttings images from a directory, sorted by depth parsed from their filenames.

    Args:
        input_dir (Path): Path to the directory containing raw cuttings photos.

    Returns:
        list[ImageMetadataCuttings]: One entry per cuttings image, sorted by depth.
    """
    # Collect all cutting images from the input directory and parse filename metadata
    imgs_metadata: list[ImageMetadataCuttings] = []
    for f in map(Path, glob.glob(str(input_dir / "*"), include_hidden=False)):
        if f.suffix.lower() in _CUTTINGS_EXTENSIONS:
            try:
                metadata = ImageMetadataCuttings.from_path(f)
                _ = metadata.shape  # validate the file is readable before segmentation runs
                imgs_metadata.append(metadata)
            except (ValueError, OSError, tifffile.TiffFileError) as e:
                logging.warning("Skipping %s: %s", f.name, e)
    imgs_metadata.sort(key=lambda m: m.depth)
    logging.info("Found %d cuttings images in %s", len(imgs_metadata), input_dir.name)

    return imgs_metadata
