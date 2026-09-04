"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from dataclasses import replace
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


def segment_full(
    img_metadata: ImageMetadataCuttings,
    config: SegmentationCuttingsConfig,
) -> CuttingsSegmentResult:
    """Segment cuttings by taking the entire image, with no cropping.

    Args:
        img_metadata (ImageMetadataCuttings): Metadata for the cuttings image to segment.
        config (SegmentationCuttingsConfig): Tunable segmentation parameters. Unused.

    Returns:
        CuttingsSegmentResult: A bbox covering the full image, and the time taken to build it.
    """
    t_start = timer()
    h, w = img_metadata.shape[:2]
    return CuttingsSegmentResult(bbox=(0, 0, w, h), time=timer() - t_start)


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
    biggest = max(props, key=lambda r: r.area) if props else None
    # a component below min_area_frac is noise, not a real pile -- treat it as "nothing found"
    # (an empty mask) so it falls through to the same full-image fallback below
    if biggest is not None and biggest.area / tray_config.work**2 >= tray_config.min_area_frac:
        m_main = lbl == biggest.label
    else:
        m_main = np.zeros_like(m_open)

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
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, tray_config.work), min(y1, tray_config.work)
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
    h, w = img.shape[:2]
    black_circle_config = config.black_circle
    gray = rgb2gray(img)
    mask = gray > black_circle_config.val_threshold
    mask = opening(mask, disk(max(1, round(black_circle_config.opening_disk * config.downscale_factor))))

    # largest connected component
    props = regionprops(label(mask))
    biggest = max(props, key=lambda p: p.area) if props else None

    # no component, or the biggest one is a handful of noise pixels rather than a real circle:
    # keep the full image rather than inscribing a square crop that means nothing
    if biggest is None or biggest.area / (h * w) < black_circle_config.min_area_frac:
        bbox = (0, 0, w, h)
    else:
        cy, cx = biggest.centroid
        r = np.sqrt(biggest.area / np.pi)
        half = int(black_circle_config.radius_shrink * r / np.sqrt(2))
        bbox = (
            max(int(cx) - half, 0),
            max(int(cy) - half, 0),
            min(int(cx) + half, w),
            min(int(cy) + half, h),
        )

    return CuttingsSegmentResult(
        bbox=scale_bbox(bbox, factor=1 / config.downscale_factor),
        time=timer() - t_start,
    )


_SEGMENTERS = {
    "full": segment_full,
    "black_circle": segment_black_circle,
    "pebble": segment_pebble,
    "tray": segment_tray,
}

DEFAULT_CUT_TYPE = "full"


def _is_full_frame_bbox(bbox: tuple[float, float, float, float], shape: tuple[int, int, int]) -> bool:
    """Whether bbox covers (approximately) the entire image, i.e. an uncropped fallback result."""
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    return x0 <= 1 and y0 <= 1 and x1 >= w - 1 and y1 >= h - 1


def _guard_degenerate_bbox(
    img_metadata: ImageMetadataCuttings, cuttings: CuttingsSegmentResult, min_crop_px: int
) -> CuttingsSegmentResult:
    """Fall back to the full image if a segmenter produced a degenerate (near-zero-size) bbox.

    A backstop shared by every segmenter, regardless of what produced the crop: a real cuttings
    region should never be a sliver a few pixels wide/tall. Catches edge cases a segmenter's own
    internal checks might miss (e.g. a candidate landing 1px short of one of its own guards)
    rather than silently producing an unusable crop.
    """
    x0, y0, x1, y1 = cuttings.bbox
    if (x1 - x0) >= min_crop_px and (y1 - y0) >= min_crop_px:
        return cuttings
    h, w = img_metadata.shape[:2]
    logger.warning(
        "Degenerate crop (%.0fx%.0f px) for %s; using the full image instead",
        x1 - x0,
        y1 - y0,
        img_metadata.image_path.name,
    )
    return replace(cuttings, bbox=(0, 0, w, h))


def _log_fallback_rate(cut_type: str, segmented: list[tuple[ImageMetadataCuttings, CuttingsSegmentResult]]) -> None:
    """Log how often this batch fell back to an uncropped result -- visible every run, not just under MLflow.

    Not a pass/fail check: there's no reliable way to tell a wrong --cut-type from a merely
    hard batch of photos, so this only surfaces the number for a human to sanity-check, never
    blocks or auto-corrects anything.
    """
    if cut_type == "full" or not segmented:
        return
    n_fallback = sum(1 for img_metadata, c in segmented if _is_full_frame_bbox(c.bbox, img_metadata.shape))
    logger.info(
        "cut_type=%s: %d/%d images (%.0f%%) fell back to an uncropped crop",
        cut_type,
        n_fallback,
        len(segmented),
        100 * n_fallback / len(segmented),
    )
    if cut_type == "pebble":
        counts = CuttingsSegmentResult.paper_status_counts([c for _, c in segmented])
        logger.info("cut_type=pebble paper detection status counts: %s", counts)


def _log_crop_size_consistency(
    cut_type: str,
    segmented: list[tuple[ImageMetadataCuttings, CuttingsSegmentResult]],
    cv_warn_threshold: float,
) -> None:
    """Log (and warn on) unusually inconsistent crop sizes for layouts that assume one fixed setup.

    black_circle/tray both assume a fairly consistent physical rig per borehole, so their
    detected crop sizes should cluster fairly tightly; a much wider spread than usual is a cheap,
    purely advisory signal that something (a mismatched --cut-type, a camera change mid-batch)
    might be off -- computed entirely from bboxes already produced, no extra image processing.
    """
    if cut_type not in ("black_circle", "tray"):
        return
    real = [
        (c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1])
        for img_metadata, c in segmented
        if not _is_full_frame_bbox(c.bbox, img_metadata.shape)
    ]
    if len(real) < 5:
        return
    widths, heights = zip(*real, strict=True)
    width_cv = float(np.std(widths) / np.mean(widths))
    height_cv = float(np.std(heights) / np.mean(heights))
    logger.info("cut_type=%s crop-size consistency: width CV=%.2f, height CV=%.2f", cut_type, width_cv, height_cv)
    if max(width_cv, height_cv) > cv_warn_threshold:
        logger.warning(
            "Unusually inconsistent %s crop sizes across the batch (width CV=%.2f, height CV=%.2f) -- "
            "double check --cut-type matches this borehole's physical layout",
            cut_type,
            width_cv,
            height_cv,
        )


def _normalize_tray_scale(cuttings: list[CuttingsSegmentResult]) -> None:
    """Set a common resize target on every tray bbox, so all trays render at the same pixel size.

    Unlike pebble/black_circle crops -- where the area outside the detected object is just
    background -- the tray bbox *is* the tray, and every tray is the same physical size, so
    differences in its detected pixel size only reflect how close the photo was taken.
    Rescaling to a shared size (rather than cropping into it) turns that pixel-size difference
    into a genuine common scale without discarding any of the tray. The target is the median
    detected size across the batch: a single far-off outlier doesn't drag every other image's
    resolution down, and only images far from the median need much up/down-sampling. Only bbox
    tuples are read here, not pixel data; the actual resize happens lazily per image when its
    crop is loaded.

    Args:
        cuttings (list[CuttingsSegmentResult]): Per-image tray detection results to set
            resize_to on, in place.
    """
    if not cuttings:
        return

    widths = sorted(c.bbox[2] - c.bbox[0] for c in cuttings)
    heights = sorted(c.bbox[3] - c.bbox[1] for c in cuttings)
    target_w = round(widths[len(widths) // 2])
    target_h = round(heights[len(heights) // 2])

    for c in cuttings:
        c.resize_to = (target_w, target_h)


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
        cut_type (str): The type of cuttings to segment: "full", "black_circle", "pebble", or "tray".
            Defaults to "full".

    Returns:
        list[ImageMetadataProcessedCuttings]: A list of processed image metadata objects. May be shorter than
        imgs_metadata if any images failed to segment.

    Raises:
        ValueError: If cut_type isn't a recognized cuttings segmentation method.
    """
    segmenter = _SEGMENTERS.get(cut_type)
    if segmenter is None:
        raise ValueError(f"Unknown cuttings type: {cut_type}")
    logger.info("Segmenting cuttings with cut_type=%s", cut_type)

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

    segmented: list[tuple[ImageMetadataCuttings, CuttingsSegmentResult]] = []
    for img_metadata in tqdm(imgs_metadata, desc="Segmenting cuttings images", mininterval=1.0):
        try:
            # reuse the shared group detection for this image's shape, if any
            shared_paper = paper_by_shape.get(img_metadata.shape)
            cuttings = shared_paper if shared_paper is not None else segmenter(img_metadata, config.cuttings)
            cuttings = _guard_degenerate_bbox(img_metadata, cuttings, config.cuttings.min_crop_px)
            segmented.append((img_metadata, cuttings))
        except (ValueError, OSError, SegmentationError) as e:
            logger.warning("Skipping %s: %s", img_metadata.image_path.name, e)

    # tray is a fixed physical size, so normalize its pixel scale across the batch before
    # building/preloading the cropped images; pebble/black_circle have no such reference
    # object, so their crops are left at their native detected size. Uncropped fallback results
    # are excluded: they're not a real tray detection, so forcing them to the tray's typical size
    # would distort the whole photo rather than leave it alone.
    if cut_type == "tray":
        real_tray_results = [
            cuttings
            for img_metadata, cuttings in segmented
            if not _is_full_frame_bbox(cuttings.bbox, img_metadata.shape)
        ]
        _normalize_tray_scale(real_tray_results)

    _log_fallback_rate(cut_type, segmented)
    _log_crop_size_consistency(cut_type, segmented, config.cuttings.crop_size_cv_warn_threshold)

    detections: list[ImageMetadataProcessedCuttings] = []
    for img_metadata, cuttings in segmented:
        detection = ImageMetadataProcessedCuttings.from_metadata(img_metadata, cuttings=cuttings, preload=cache)
        detections.append(detection)

        if with_mlflow and debug:
            log_image_metadata_processed_cuttings_mlflow(
                result=detection,
                filename=f"{img_metadata.image_path.stem}",
                subfolder="debug",
            )

    if with_mlflow:
        log_cuttings_segmentation_results_with_mlflow(detections, time=timer() - t_start)

    return detections
