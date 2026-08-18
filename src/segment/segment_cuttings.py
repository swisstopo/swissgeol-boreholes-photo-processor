"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening
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
    mask = (hsv[..., 2] > 0.85) & (hsv[..., 1] < 0.15)  # bright AND colorless

    # the black stripes punch holes in the paper — close them
    mask = closing(mask, disk(max(1, round(25 * config.downscale_factor))))

    lbl = label(mask)
    props = regionprops(lbl)

    # the paper is a solid rectangle -- it should fill nearly all of its own
    # bounding box (extent) and be close to convex (solidity). A merged blob of
    # stones is porous (lots of black gaps between pebbles survive closing) and
    # scores much lower on both, even when its raw area is larger. The paper is
    # always anchored to an edge of the *original* photo's bottom, but load_image
    # rotates portrait photos 90 degrees to landscape, which moves that edge to the
    # right rather than the bottom -- so a candidate is only trustworthy if it
    # reaches the bottom OR the right edge, no matter how high its extent/solidity.
    # It's also always a modest slice of the frame; a blob covering most of the
    # image (e.g. a wash of pale sand/rock filling most of the shot, which can
    # coincidentally touch both edges) is never the paper regardless of shape score.
    MIN_EXTENT = 0.45
    MIN_SOLIDITY = 0.7
    MAX_AREA_FRAC = 0.35
    EDGE_MARGIN = 0.03  # fraction of the relevant dimension the candidate must reach to count as edge-anchored

    def touches_bottom(p):
        return p.bbox[2] >= (1 - EDGE_MARGIN) * h

    def touches_right(p):
        return p.bbox[3] >= (1 - EDGE_MARGIN) * w

    candidates = [
        p
        for p in props
        if p.extent >= MIN_EXTENT
        and p.solidity >= MIN_SOLIDITY
        and p.area <= MAX_AREA_FRAC * h * w
        and (touches_bottom(p) or touches_right(p))
    ]
    paper = max(candidates, key=lambda p: p.area) if candidates else None

    # crop away the side of the frame the paper sits on, preferring the bottom
    # when it touches both (the common corner case). A degenerate paper region
    # (spanning the full height or width) usually just means the reference sheet
    # isn't reliably in frame, so the whole image is already the cuttings region.
    bbox = (0, 0, w, h)
    if paper is not None:
        if touches_bottom(paper) and paper.bbox[0] > 0:
            bbox = (0, 0, w, paper.bbox[0])
        elif touches_right(paper) and paper.bbox[1] > 0:
            bbox = (0, 0, paper.bbox[1], h)
        else:
            paper = None

    if paper is None:
        logger.warning(
            "No reliable paper region found for %s; using the full image as the cuttings region",
            img_metadata.image_path.name,
        )

    return CuttingsSegmentResult(
        bbox=scale_bbox(bbox, factor=1 / config.downscale_factor),
        time=timer() - t_start,
    )


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
