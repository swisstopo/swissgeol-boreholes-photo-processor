"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from timeit import default_timer as timer

from tqdm import tqdm

from src.config import SegmentationConfig, SegmentationError
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
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting cuttings images"):
        try:
            height, width, _ = img_metadata.shape
            cuttings = CuttingsSegmentResult(bbox=(0, 0, width, height))
            processed_metadata = ImageMetadataProcessedCuttings.from_metadata(img_metadata, cuttings=cuttings)
            detections.append(processed_metadata)

            if with_mlflow and debug:
                log_image_metadata_processed_cuttings_mlflow(
                    result=processed_metadata,
                    filename=f"{img_metadata.image_path.stem}",
                    subfolder="debug",
                )
        except (ValueError, OSError, SegmentationError) as e:
            logger.warning("Skipping %s: %s", img_metadata.filename, e)

    if with_mlflow:
        log_cuttings_segmentation_results_with_mlflow(detections, time=timer() - t_start)

    return detections
