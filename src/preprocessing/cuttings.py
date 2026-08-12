"""Module for preprocessing the cuttings images."""

import glob
import logging
from collections import defaultdict
from pathlib import Path

import tifffile

from src.config import SegmentationError
from src.mlflow_utils import log_collect_cuttings_results_with_mlflow
from src.models import ImageMetadataCuttings

_CUTTINGS_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_cuttings(input_dir: Path, with_mlflow: bool = False) -> list[ImageMetadataCuttings]:
    """Collect cuttings images from a directory, sorted by depth parsed from their filenames.

    Only the first image (by filename) at each depth is kept; the rest are dropped as
    duplicates and their count is logged to MLflow when with_mlflow is set. "00-Vials-"
    files (e.g. GVL-1's sample-vial photos) are excluded outright: their depth-less names
    would otherwise parse as depth 0 and pollute the output.

    Args:
        input_dir (Path): Path to the directory containing raw cuttings photos.
        with_mlflow (bool): Whether to log duplicate-depth stats to MLflow.

    Returns:
        list[ImageMetadataCuttings]: One entry per depth, sorted by depth.
    """
    # Collect all cutting images from the input directory and parse filename metadata
    imgs_metadata: list[ImageMetadataCuttings] = []
    for f in map(Path, glob.glob(str(input_dir / "*"), include_hidden=False)):
        if f.name.lower().startswith("00-vials-"):
            continue
        if f.suffix.lower() in _CUTTINGS_EXTENSIONS:
            try:
                metadata = ImageMetadataCuttings.from_path(f)
                metadata.borehole_id = input_dir.name
                _ = metadata.shape  # validate the file is readable before segmentation runs
                imgs_metadata.append(metadata)
            except (ValueError, OSError, tifffile.TiffFileError, SegmentationError) as e:
                logging.warning("Skipping %s: %s", f.name, e)
    imgs_metadata.sort(key=lambda m: (m.depth, m.image_path.name))

    deduped_metadata: list[ImageMetadataCuttings] = []
    duplicate_counts: dict[float, int] = defaultdict(int)
    seen_depths: set[float] = set()
    for metadata in imgs_metadata:
        if metadata.depth in seen_depths:
            duplicate_counts[metadata.depth] += 1
            continue
        seen_depths.add(metadata.depth)
        deduped_metadata.append(metadata)

    if duplicate_counts:
        logging.warning(
            "Dropped %d duplicate-depth cuttings image(s) in %s", sum(duplicate_counts.values()), input_dir.name
        )
    if with_mlflow:
        log_collect_cuttings_results_with_mlflow(duplicate_counts)

    logging.info("Found %d cuttings images in %s", len(deduped_metadata), input_dir.name)

    return deduped_metadata
