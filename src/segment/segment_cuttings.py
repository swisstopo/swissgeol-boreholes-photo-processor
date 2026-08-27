"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from timeit import default_timer as timer

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import scharr, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion, opening
from skimage.transform import resize
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
from src.segment.utils.cuttings import ProcessPebblePaperGroupByShape, detect_paper, resolve_paper_crop
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def segment_tray(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings that are inside a tray, via an edge-density quantile bounding box.

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
    tray_config = config.tray

    # texture energy
    g = rgb2gray(resize(img, (tray_config.work, tray_config.work), anti_aliasing=True))  # float in [0,1]
    grad = scharr(g)  # gradient magnitude
    energy = uniform_filter(grad, size=33)  # 33x33 local mean

    # otsu mask
    otsu_t = threshold_otsu(energy)
    m = energy > otsu_t

    # keep only the largest connected component: a printed label/tag has its own
    # high edge-density text, so it can show up as a second, disconnected blob in
    # the mask, and taking quantiles over all mask pixels would pull the box
    # toward the label instead of the tray. `opening` first drops thin
    # bridges/specks so the label can't be connected to the pile through a noisy
    # sliver, then we keep only the single biggest component (assumed to be the
    # tray/pile) before computing quantiles.
    m_open = opening(m, disk(tray_config.open_radius))
    lbl = label(m_open)
    props = regionprops(lbl)
    m_main = (lbl == max(props, key=lambda r: r.area).label) if props else m

    # the local-mean energy smoothing above bleeds a sliver of the mask past the true edge onto
    # the surrounding tray/table; erode it back before taking the bbox
    m_main = erosion(m_main, disk(tray_config.erosion_radius))

    ys, xs = np.nonzero(m_main)
    if len(xs) == 0:
        bbox = (0, 0, w, h)
    else:
        q = (1 - tray_config.coverage**0.5) / 2
        x0, x1 = np.quantile(xs, [q, 1 - q])
        y0, y1 = np.quantile(ys, [q, 1 - q])
        if tray_config.square:
            s = max(x1 - x0, y1 - y0) / 2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1, y0, y1 = cx - s, cx + s, cy - s, cy + s
        bbox = (
            x0 * w / tray_config.work,
            y0 * h / tray_config.work,
            x1 * w / tray_config.work,
            y1 * h / tray_config.work,
        )

    return CuttingsSegmentResult(
        bbox=scale_bbox(bbox, factor=1 / config.downscale_factor),
        time=timer() - t_start,
    )


def segment_pebble(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment pebble cuttings laid out above a reference paper sheet.

    Args:
        img_metadata (ImageMetadataCuttings): Metadata for the cuttings image to segment.
        config (SegmentationCuttingsConfig): Tunable segmentation parameters.

    Returns:
        CuttingsSegmentResult: The bounding box of the cuttings region, the time taken to
        segment it, and the PaperDetectionStatus outcome of the paper-sheet detection.
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

    status, bbox = resolve_paper_crop(paper, h, w, pebble_config.max_cropped_frac)

    return CuttingsSegmentResult(
        bbox=scale_bbox(bbox, factor=1 / config.downscale_factor),
        time=timer() - t_start,
        paper_status=status,
    )


def segment_black_circle(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings that are inside a black circle.

    Args:
        img_metadata (ImageMetadataCuttings): Metadata for the cuttings image to segment.
        config (SegmentationCuttingsConfig): Tunable segmentation parameters.

    Returns:
        CuttingsSegmentResult: The bounding box of the cuttings region and the time taken to
        segment it.
    """
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
    "tray": segment_tray,
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
        cut_type (str): The type of cuttings to segment: "black_circle", "pebble", or "tray".
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

    # Pebble cuttings share a physical layout (a reference paper sheet) with the rest of a
    # same-shape batch far more often than per-image thresholding alone can reliably tell --
    # estimate it once per shape group and reuse it, falling back to per-image detection for
    # images whose shape group is too small (or inconsistent) to trust a shared estimate.
    paper_by_shape: dict[tuple[int, int, int], CuttingsSegmentResult] = {}
    if cut_type == "pebble":
        logger.info("Processing pebble paper regions by group ...")
        paper_by_shape = ProcessPebblePaperGroupByShape(config.cuttings.pebble_group, config.n_workers).run(
            imgs_metadata
        )

    detections: list[ImageMetadataProcessedCuttings] = []
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting cuttings images", mininterval=1.0):
        try:
            # segmentation: reuse the shared group detection for this image's shape, if any
            shared_paper = paper_by_shape.get(img_metadata.shape)
            cuttings = shared_paper if shared_paper is not None else segmenter(img_metadata, config.cuttings)
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
