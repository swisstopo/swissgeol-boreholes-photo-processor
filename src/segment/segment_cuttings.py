"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening
from tqdm import tqdm

from src.config import SegmentationConfig, SegmentationCuttingsConfig, SegmentationError
from src.mlflow_utils import (
    log_cuttings_segmentation_results_with_mlflow,
    log_image_metadata_processed_cuttings_mlflow,
)
from src.models import (
    CuttingsSegmentResult,
    ImageMetadataCuttings,
    ImageMetadataProcessedCuttings,
)

logger = logging.getLogger(__name__)


# TODO: add params to the config


def segment_black_circle(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings that are inside a black circle."""
    t_start = timer()
    img = img_metadata.load_image()
    gray = rgb2gray(img)
    mask = gray > 0.16
    mask = opening(mask, disk(7))

    # largest connected component
    lbl = label(mask)
    props = regionprops(lbl)
    biggest = max(props, key=lambda p: p.area)

    cy, cx = biggest.centroid
    r = np.sqrt(biggest.area / np.pi)
    half = int(0.98 * r / np.sqrt(2))

    return CuttingsSegmentResult(
        bbox=(
            max(int(cx) - half, 0),
            max(int(cy) - half, 0),
            int(cx) + half,
            int(cy) + half,
        ),
        time=timer() - t_start,
    )


def segment_cuttings(
    imgs_metadata: list[ImageMetadataCuttings],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
    debug: bool = False,
) -> list[ImageMetadataProcessedCuttings]:
    """Segment the input images and return a list of processed image metadata objects.

    TODO: Placeholder function; the cuttings bbox is currently the entire image rather
    than a detected region.

    Args:
        imgs_metadata (list[ImageMetadataCuttings]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.
        debug (bool): Whether to additionally log each image's cuttings bbox overlay to MLflow.
            Only applies when with_mlflow is True.

    Returns:
        list[ImageMetadataProcessedCuttings]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.
    """
    config = config or SegmentationConfig()
    t_start = timer()

    detections: list[ImageMetadataProcessedCuttings] = []
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting cuttings images", mininterval=1.0):
        try:
            # rotate image if its not landscape

            # segmentation
            cuttings = segment_black_circle(img_metadata, config.cuttings)
            detection = ImageMetadataProcessedCuttings.from_metadata(img_metadata, cuttings=cuttings)
            detections.append(detection)

            # tracking
            if with_mlflow and debug:
                log_image_metadata_processed_cuttings_mlflow(
                    result=detection,
                    filename=f"{img_metadata.image_path.stem}",
                    subfolder="debug",
                )

        except (ValueError, OSError, SegmentationError) as e:
            logger.warning("Skipping %s: %s", img_metadata.image_path.name, e)
            continue

    if with_mlflow:
        log_cuttings_segmentation_results_with_mlflow(detections, time=timer() - t_start)

    return detections
