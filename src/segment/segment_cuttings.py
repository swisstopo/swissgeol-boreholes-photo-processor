"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
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
    PaperDetectionStatus,
)
from src.segment.utils.cuttings import detect_paper
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def segment_pebble(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment pebble cuttings laid out above a reference paper sheet.

    Args:
        img_metadata (ImageMetadataCuttings): Metadata for the cuttings image to segment.
        config (SegmentationCuttingsConfig): Tunable segmentation parameters.

    Returns:
        CuttingsSegmentResult: The bounding box of the cuttings region and the time taken to
        segment it.
    """
    t_start = timer()
    img = img_metadata.load_image(factor=config.downscale_factor)
    h, w = img.shape[:2]
    hsv = rgb2hsv(img)
    pebble_config = config.pebble

    # fixed threshold first; some images are shot at a much darker exposure and
    # never produce a usable candidate there, so retry with a much looser
    # brightness cutoff -- still "bright relative to the surrounding rock", just
    # not absolute-white
    paper = detect_paper(
        hsv, h, w, pebble_config.val_threshold_strict, config.downscale_factor, pebble_config
    ) or detect_paper(hsv, h, w, pebble_config.val_threshold_loose, config.downscale_factor, pebble_config)

    if paper is None:
        status = PaperDetectionStatus.NO_CANDIDATE

    # the paper always sits toward the bottom-right of the frame, so the cuttings
    # region is always everything to the left of its left edge -- never crop by
    # height, even when the candidate was accepted for touching the bottom rather
    # than the right. A degenerate paper region (left edge at column 0, i.e.
    # nothing left to keep) usually just means the reference sheet isn't reliably
    # in frame, so the whole image is already the cuttings region. The paper is
    # also always a modest slice of the frame, so a candidate that would crop away
    # more than half the image is more likely a misdetection than a real card.
    bbox = (0, 0, w, h)
    if paper is not None:
        if paper.bbox[1] == 0:
            status = PaperDetectionStatus.DEGENERATE_LEFT_EDGE
            paper = None
        elif (w - paper.bbox[1]) > pebble_config.max_cropped_frac * w:
            status = PaperDetectionStatus.CROPPED_TOO_MUCH
            paper = None
        else:
            bbox = (0, 0, paper.bbox[1], h)
            status = PaperDetectionStatus.FOUND

    if paper is None:
        logger.warning(
            "No reliable paper region found for %s (%s); using the full image as the cuttings region",
            img_metadata.image_path.name,
            status.value,
        )

    return CuttingsSegmentResult(
        bbox=scale_bbox(bbox, factor=1 / config.downscale_factor),
        time=timer() - t_start,
        paper_status=status,
    )


def segment_black_circle(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings that are inside a black circle."""
    t_start = timer()
    img = img_metadata.load_image(factor=config.downscale_factor)
    black_circle_config = config.black_circle
    gray = rgb2gray(img)
    mask = gray > black_circle_config.val_threshold
    mask = opening(mask, disk(max(1, round(black_circle_config.opening_disk * config.downscale_factor))))

    # largest connected component
    lbl = label(mask)
    props = regionprops(lbl)
    biggest = max(props, key=lambda p: p.area)

    cy, cx = biggest.centroid
    r = np.sqrt(biggest.area / np.pi)
    half = int(black_circle_config.radius_shrink * r / np.sqrt(2))

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
    "pebble": segment_pebble,
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
        cut_type (str): The type of cuttings to segment: "black_circle" or "pebble".
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
