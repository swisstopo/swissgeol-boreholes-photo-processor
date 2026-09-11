"""Entry point for segmenting a batch of borehole cuttings images."""

import logging
from collections import defaultdict
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
    resized_gray = rgb2gray(resize(img, (tray_config.work, tray_config.work), anti_aliasing=True))  # float in [0,1]
    grad = scharr(resized_gray)  # gradient magnitude
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


def log_fallback_rate(cut_type: str, segmented: list[tuple[ImageMetadataCuttings, CuttingsSegmentResult]]) -> None:
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


def log_crop_size_consistency(
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


def _normalize_tray_size(
    cuttings: list[CuttingsSegmentResult],
    target_w: float | None = None,
    target_h: float | None = None,
    max_scale_factor: float = 2.0,
) -> list[CuttingsSegmentResult]:
    """Bring every tray crop to a shared (width, height), cropping first so nothing is ever stretched.

    First crops (never stretches) whichever axis is in excess so the crop's aspect ratio matches
    the target's; from there, reaching the exact target size is always a uniform scale (upsample
    if still smaller, guarded by max_scale_factor; crop further if still bigger, unguarded, since
    cropping doesn't degrade the pixels it keeps). Only bbox tuples are read here; the actual
    resize happens lazily when a crop is loaded.

    Args:
        cuttings (list[CuttingsSegmentResult]): Per-image tray detections to normalize.
        target_w (float | None): Target width; defaults to the median width across `cuttings`.
        target_h (float | None): Target height; defaults to the median height across `cuttings`.
        max_scale_factor (float): Max factor an aspect-matched crop may be upsampled by before
            it's left at that native size instead of resampling it further.

    Returns:
        list[CuttingsSegmentResult]: Copies with resize_to set (upsampled), bbox trimmed
        (cropped), or left at the aspect-matched native size (upsample guard triggered).
    """
    if not cuttings:
        return []

    if target_w is None:
        widths = sorted(c.bbox[2] - c.bbox[0] for c in cuttings)
        target_w = widths[len(widths) // 2]
    if target_h is None:
        heights = sorted(c.bbox[3] - c.bbox[1] for c in cuttings)
        target_h = heights[len(heights) // 2]
    target_w = round(target_w)
    target_h = round(target_h)
    target_ratio = target_w / target_h

    normalized = []
    for c in cuttings:
        x0, y0, x1, y1 = c.bbox
        width = x1 - x0
        height = y1 - y0
        ratio = width / height

        if ratio > target_ratio:
            width = height * target_ratio
            x1 = x0 + width
        elif ratio < target_ratio:
            height = width / target_ratio
            y1 = y0 + height

        scale_factor = target_w / width  # == target_h / height, since the ratio now matches
        if scale_factor > 1:
            if scale_factor > max_scale_factor:
                logger.warning(
                    "Tray crop would need to be upsampled %.2fx to match the target size; "
                    "leaving it at its native (aspect-matched) size instead of resampling it that far",
                    scale_factor,
                )
                normalized.append(replace(c, bbox=(x0, y0, x1, y1)))
                continue
            normalized.append(replace(c, resize_to=(target_w, target_h), bbox=(x0, y0, x1, y1)))
        elif scale_factor < 1:
            normalized.append(replace(c, bbox=(x0, y0, x0 + target_w, y0 + target_h)))
        else:
            normalized.append(replace(c, bbox=(x0, y0, x1, y1)))
    return normalized


def _cluster_shape_group_target_widths(
    median_width_by_shape: dict[tuple[int, int, int], float], merge_tolerance: float
) -> dict[tuple[int, int, int], float]:
    """Merge native shape groups whose median widths are within merge_tolerance, onto the smaller.

    Different camera resolutions can still land on almost the same tray width once each group is
    normalized on its own -- that's detection noise, not a genuinely different tray size. Shape
    groups are sorted by median and chained onto the current cluster's smallest member if within
    tolerance of it (not just the previous neighbor, so a cluster can't drift arbitrarily far).

    Args:
        median_width_by_shape (dict[tuple[int, int, int], float]): Each shape group's own median width.
        merge_tolerance (float): Max relative difference from a cluster's smallest member for a
            group to join it.

    Returns:
        dict[tuple[int, int, int], float]: Target width per shape group -- its own median, or its
        cluster's smallest median if merged.
    """
    if not median_width_by_shape:
        return {}

    ordered = sorted(median_width_by_shape.items(), key=lambda kv: kv[1])
    target_by_shape: dict[tuple[int, int, int], float] = {}
    cluster_min = ordered[0][1]
    for shape, median_width in ordered:
        if (median_width - cluster_min) / cluster_min > merge_tolerance:
            cluster_min = median_width
        target_by_shape[shape] = cluster_min
    return target_by_shape


def _cluster_shape_group_target_heights(
    target_width_by_shape: dict[tuple[int, int, int], float],
    heights_by_shape: dict[tuple[int, int, int], list[float]],
) -> dict[tuple[int, int, int], float]:
    """Share a pooled target height across shape groups already merged (by width) onto one cluster.

    Reuses the width-based cluster membership (shape groups sharing a target width are the same
    cluster) and takes the median over every individual image's height pooled across the whole
    cluster -- not one shape group's own median -- so a single small/atypical group can't drag
    the shared height for the rest of the cluster.

    Args:
        target_width_by_shape (dict[tuple[int, int, int], float]): Output of _cluster_shape_group_target_widths.
        heights_by_shape (dict[tuple[int, int, int], list[float]]): Each shape group's own per-image heights.

    Returns:
        dict[tuple[int, int, int], float]: Target height per shape group, shared within cluster.
    """
    pooled_heights_by_cluster: dict[float, list[float]] = defaultdict(list)
    for shape, target_w in target_width_by_shape.items():
        pooled_heights_by_cluster[target_w].extend(heights_by_shape[shape])
    target_height_by_cluster = {
        target_w: sorted(heights)[len(heights) // 2] for target_w, heights in pooled_heights_by_cluster.items()
    }
    return {shape: target_height_by_cluster[target_w] for shape, target_w in target_width_by_shape.items()}


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

    # Tray is a fixed physical size, so every crop's (width, height) is normalized to a shared
    # target before preloading -- computed per native image shape (different camera resolutions
    # shouldn't share one median), then shape groups whose medians land close enough are merged
    # onto the smaller one. Uncropped fallback results are excluded, since they aren't real
    # detections. Pebble/black_circle have no such reference object and are left untouched.
    if cut_type == "tray":
        real_tray_indices_by_shape: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, (img_metadata, cuttings) in enumerate(segmented):
            if not _is_full_frame_bbox(cuttings.bbox, img_metadata.shape):
                real_tray_indices_by_shape[img_metadata.shape].append(i)

        median_width_by_shape = {
            shape: sorted(segmented[i][1].bbox[2] - segmented[i][1].bbox[0] for i in shape_indices)[
                len(shape_indices) // 2
            ]
            for shape, shape_indices in real_tray_indices_by_shape.items()
        }
        heights_by_shape = {
            shape: [segmented[i][1].bbox[3] - segmented[i][1].bbox[1] for i in shape_indices]
            for shape, shape_indices in real_tray_indices_by_shape.items()
        }
        target_width_by_shape = _cluster_shape_group_target_widths(
            median_width_by_shape, config.cuttings.tray.shape_group_merge_tolerance
        )
        target_height_by_shape = _cluster_shape_group_target_heights(target_width_by_shape, heights_by_shape)

        for shape, shape_indices in real_tray_indices_by_shape.items():
            normalized_cuttings = _normalize_tray_size(
                [segmented[i][1] for i in shape_indices],
                target_w=target_width_by_shape[shape],
                target_h=target_height_by_shape[shape],
                max_scale_factor=config.cuttings.tray.max_scale_factor,
            )
            for i, cuttings in zip(shape_indices, normalized_cuttings, strict=True):
                segmented[i] = (segmented[i][0], cuttings)

    log_fallback_rate(cut_type, segmented)
    log_crop_size_consistency(cut_type, segmented, config.cuttings.crop_size_cv_warn_threshold)

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
