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
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def segment_black_circle(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings that are inside a black circle."""
    t_start = timer()
    img = img_metadata.load_image(factor=config.downscale_factor)
    gray = rgb2gray(img)
    mask = gray > 0.16
    mask = opening(mask, disk(max(1, round(7 * config.downscale_factor))))

    # largest connected component
    lbl = label(mask)
    props = regionprops(lbl)
    biggest = max(props, key=lambda p: p.area)

    cy, cx = biggest.centroid
    r = np.sqrt(biggest.area / np.pi)
    half = int(0.98 * r / np.sqrt(2))

    return CuttingsSegmentResult(
        bbox=scale_bbox(
            (
                max(int(cx) - half, 0),
                max(int(cy) - half, 0),
                min(int(cx) + half, img.shape[1]),
                min(int(cy) + half, img.shape[0]),
            ),
            factor=1 / config.downscale_factor,
        ),
        time=timer() - t_start,
    )


_SEGMENTERS = {
    "black_circle": segment_black_circle,
}

DEFAULT_CUT_TYPE = "black_circle"


def segment_cuttings(
    imgs_metadata: list[ImageMetadataCuttings],
    config: SegmentationConfig | None = None,
    with_mlflow: bool = False,
    debug: bool = False,
    cache: bool = False,
    cut_type: str = DEFAULT_CUT_TYPE,
) -> list[ImageMetadataProcessedCuttings]:
    """Segment the input images and return a list of processed image metadata objects.

    Args:
        imgs_metadata (list[ImageMetadataCuttings]): A list of image metadata objects to be segmented.
        config (SegmentationConfig | None): Tunable segmentation parameters. Defaults to SegmentationConfig().
        with_mlflow (bool): Whether to log artifacts to MLflow.
        debug (bool): Whether to additionally log each image's cuttings bbox overlay to MLflow.
            Only applies when with_mlflow is True.
        cache (bool): Whether to eagerly load and cache each image's cropped region in memory.
        cut_type (str): The type of cuttings to segment: "black_circle".
            Defaults to "black_circle".

    Returns:
        list[ImageMetadataProcessedCuttings]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.

    Raises:
        ValueError: If cut_type isn't a recognized cuttings segmentation method.
    """
    segmenter = _SEGMENTERS.get(cut_type)
    if segmenter is None:
        raise ValueError(f"Unknown cuttings type: {cut_type}")

    config = config or SegmentationConfig()
    t_start = timer()

    detections: list[ImageMetadataProcessedCuttings] = []
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting cuttings images", mininterval=1.0):
        try:
            # segmentation
            cuttings = segmenter(img_metadata, config.cuttings)
            detection = ImageMetadataProcessedCuttings.from_metadata(img_metadata, cuttings=cuttings, preload=cache)
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

    if with_mlflow:
        log_cuttings_segmentation_results_with_mlflow(detections, time=timer() - t_start)

    return detections
