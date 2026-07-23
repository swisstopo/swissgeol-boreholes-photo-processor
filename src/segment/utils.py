"""Helper functions for image segmentation."""

import logging
from itertools import groupby

import numpy as np
import pytesseract
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gaussian, threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.config import (
    SegmentationCoreConfig,
    SegmentationError,
    SegmentationRulerConfig,
    SegmentationTrayMultipleConfig,
    SegmentationTraySingleConfig,
)
from src.models import ImageMetadata, ImageSegmentResult, RulerSegmentResult
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def segment_ruler(img_metadata: ImageMetadata, config: SegmentationRulerConfig) -> RulerSegmentResult | None:
    """Detect a depth ruler by OCR'ing its printed number ticks and derive a pixel-to-unit scale.

    Binarizes the image for more reliable OCR, keeps digit detections within
    [text_min_value, text_max_value], and drops detections whose spacing deviates
    from the median step between consecutive numbers (outliers).

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and segment.
        config (SegmentationRulerConfig): Tunable segmentation parameters.

    Returns:
        RulerSegmentResult | None: Bounding box enclosing all detected ruler numbers, the
            pixel-per-unit scale, and the per-number bounding boxes, or None if no ruler
            numbers were detected.
    """
    img = img_metadata.load_image(factor=config.downscale_factor)

    # OCR performs better on binarized images
    img_gray = rgb2gray(img)
    local_thresh = threshold_otsu(img_gray)
    img_bin = (img_gray > local_thresh).astype(np.uint8)

    # Run OCR
    img_data = pytesseract.image_to_data(255 * img_bin, output_type=pytesseract.Output.DICT)

    data = np.array(
        # Only keep text from text_min_value to text_max_value
        [
            (int(text), left, top, width, height)
            for text, left, top, width, height in zip(
                img_data["text"], img_data["left"], img_data["top"], img_data["width"], img_data["height"], strict=True
            )
            if text.isdigit() and config.text_min_value <= int(text) <= config.text_max_value
        ]
    )

    if data.size == 0:
        return None

    # Central point of detected number (left + width/2, top + height / 2)
    X = data[:, [1, 2]] + data[:, [3, 4]] / 2
    y = data[:, 0]

    # Sort values in increasing order and compute steps / median step (robust to outliers)
    y_sort = np.argsort(y)
    X_diff = np.linalg.norm(np.diff(X[y_sort], axis=0), axis=1)
    y_diff = np.diff(y[y_sort], axis=0)
    steps_median = np.median(X_diff[y_diff != 0] / y_diff[y_diff != 0]).item()

    # Drop detections that are not aligned with detected steps (distance to neighbor)
    distances = pairwise_distances(X) / (pairwise_distances(y[:, None]) + 1e-16)
    distances_idx = ~np.eye(distances.shape[0], dtype=bool)
    distances = distances[distances_idx].reshape(
        (
            distances.shape[0],
            distances.shape[1] - 1,
        )
    )  # Remove diagonal NxN -> Nx(N-1)
    id_inliers = abs(np.median(distances, axis=1) - steps_median) / steps_median < config.r_error_outliers

    if not id_inliers.any():
        return None

    # Reconstruct bbox for each unit (left, top, left + width, top + height)
    bbox_units = np.concatenate(
        (
            data[id_inliers][:, [1, 2]],
            data[id_inliers][:, [1, 2]] + data[id_inliers][:, [3, 4]],
        ),
        axis=1,
    )
    bbox_units = (1 / config.downscale_factor) * bbox_units

    return RulerSegmentResult(
        bbox=(
            bbox_units[:, 0].min().item(),
            bbox_units[:, 1].min().item(),
            bbox_units[:, 2].max().item(),
            bbox_units[:, 3].max().item(),
        ),
        px_per_unit=(1 / config.downscale_factor) * steps_median,
        bbox_units=[tuple(row) for row in bbox_units.tolist()],
    )


def segment_ruler_by_group(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationRulerConfig,
) -> dict[tuple[int, int, int], RulerSegmentResult | None]:
    """Detect the depth ruler once per group of same-shaped images, aggregated over several images.

    Images sharing a shape are assumed to come from the same static camera setup, so the ruler
    position and scale are shared too. OCR is run on images within a shape group, in order,
    until config.n_min_ruler detections succeed (or the group is exhausted). The median
    detection (by px_per_unit) is reused for every image in the group, which is more robust to
    per-image OCR noise (lighting, thresholding) than trusting a single detection.

    Args:
        imgs_metadata (list[ImageMetadata]): All images to consider, potentially spanning
            multiple shapes.
        config (SegmentationRulerConfig): Tunable segmentation parameters.

    Returns:
        dict[tuple[int, int, int], RulerSegmentResult | None]: Detected ruler per image
            shape, or None for groups where no image yielded a ruler detection.
    """
    groups = group_images_by_shape(imgs_metadata)
    results: dict[tuple[int, int, int], RulerSegmentResult | None] = {}
    for shape, group in groups.items():
        detections: list[RulerSegmentResult] = []
        for img_metadata in group:
            if len(detections) >= config.n_min_ruler:
                break
            try:
                detection = segment_ruler(img_metadata, config)
            except SegmentationError as e:
                logger.warning("%s. Skipping.", e)
                continue
            if detection is not None:
                detections.append(detection)

        results[shape] = _aggregate_ruler_detections(detections)

    logger.info("Computed shared ruler for %d/%d shape group(s).", sum(1 for r in results.values() if r), len(groups))

    return results


def _aggregate_ruler_detections(detections: list[RulerSegmentResult]) -> RulerSegmentResult | None:
    """Pick the median-by-scale detection among several, to be robust to per-image OCR noise.

    Args:
        detections (list[RulerSegmentResult]): Independent ruler detections for images assumed
            to share the same ruler position and scale.

    Returns:
        RulerSegmentResult | None: The detection whose px_per_unit is the median of the batch,
            or None if no detections were given.
    """
    if not detections:
        return None
    return sorted(detections, key=lambda d: d.px_per_unit)[len(detections) // 2]


def segment_tray_single(img_metadata: ImageMetadata, config: SegmentationTraySingleConfig) -> ImageSegmentResult:
    """Segment a single image via thresholding when no shared foreground bbox is available.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and segment.
        config (SegmentationTraySingleConfig): Tunable segmentation parameters.

    Returns:
        ImageSegmentResult: Bounding box as (x_min, y_min, x_max, y_max), in the
            original image's coordinate space.
    """
    factor = config.downscale_factor
    binary, grey = _apply_threshold_and_clean(
        img_metadata.load_image(factor=factor),
        min_object_size=max(1, round(config.min_object_size * factor**2)),  # factor**2 for area-based configs
        opening_disk=max(1, round(config.opening_disk * factor)),
        closing_disk=max(1, round(config.closing_disk * factor)),
    )

    bbox = _select_bbox(
        img_mask=binary,
        img_intensity=grey,
        img_height=binary.shape[0],
        min_bbox_height=max(1, round(config.min_bbox_height * factor)),
        edge_margin_top=round(config.edge_margin_top * factor),
        edge_margin_bottom=round(config.edge_margin_bottom * factor),
        min_size_for_bottom=round(config.min_size_for_bottom * factor**2),
    )

    return ImageSegmentResult(bbox=scale_bbox(bbox, factor=1 / factor))


def segment_tray_multiple(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationTrayMultipleConfig,
) -> ImageSegmentResult | None:
    """Estimate the foreground mask shared by a batch of images taken from a static camera position.

    Loads and downscales every image, then computes the per-pixel standard deviation across the
    stack: pixels that change between shots (the core) get a high value, while the static
    background (tray, backdrop) stays low.

    Args:
        imgs_metadata (list[ImageMetadata]): Batch of images assumed to share the same camera position.
        config (SegmentationTrayMultipleConfig): Tunable segmentation parameters.

    Returns:
        ImageSegmentResult | None: Estimated tray bounding box, or None if unavailable.
    """
    imgs_stack: list[np.ndarray] = []
    img_shape: tuple[int, ...] | None = None
    for img_metadata in tqdm(imgs_metadata, desc="Load images"):
        try:
            img = img_metadata.load_image(factor=config.downscale_factor)
        except (SegmentationError, ValueError) as e:
            logger.warning("%s. Skipping.", e)
            continue

        if img_shape is None:
            img_shape = img.shape
        elif img_shape != img.shape:
            logger.warning("Inconsistent image sizes")
            return None

        img_gray_ = rgb2gray(img)
        img_blur_ = gaussian(img_gray_, sigma=config.foreground_blur_sigma)
        imgs_stack.append(img_blur_)

    # At least n_min images for statistics
    if len(imgs_stack) < config.n_min_foreground:
        return None

    # Compute STD between images to highlight changes in background
    fg_img = np.stack(imgs_stack).std(axis=0)
    fg_bbox = _estimate_tray_bbox(fg_img)
    if fg_bbox is None:
        return None

    return ImageSegmentResult(bbox=scale_bbox(fg_bbox, factor=1 / config.downscale_factor))


def _estimate_tray_bbox(fg_img: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Fit the foreground distribution and derive a bounding box for the core region.

    Assumes the foreground shows the highest variance. Fits a 2-component GMM over the
    per-pixel values and selects the component with the highest mean as foreground. The
    foreground mask threshold is set to mu - std of that component, and the largest
    connected region in the resulting mask is taken as the core.

    Args:
        fg_img (np.ndarray | None): Foreground distribution map or None if unavailable.

    Returns:
        tuple[int, int, int, int] | None: Bounding box as (x_min, y_min, x_max, y_max), or None.
    """
    if fg_img is None:
        return None

    # Fit GMM to get background and foreground distributions
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(fg_img.flatten().reshape(-1, 1))
    means = np.asarray(gmm.means_)
    covariances = np.asarray(gmm.covariances_)
    fg_id = np.argmax(means)
    fg_mask = fg_img > means[fg_id] - np.sqrt(covariances[fg_id])

    # Foreground is defined as the largest connected region (area)
    props = regionprops(label(fg_mask), intensity_image=fg_img)

    if not props:
        return None

    props = sorted(props, key=lambda x: x.area, reverse=True)
    bbox = props[0].bbox
    return (bbox[1], bbox[0], bbox[3] - 1, bbox[2] - 1)


def _apply_threshold_and_clean(
    img: np.ndarray,
    min_object_size: int,
    opening_disk: int,
    closing_disk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholding to the input image and return a cleaned binary mask.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be thresholded.
        min_object_size (int): Minimum size of objects to be retained.
        opening_disk (int): Size of the disk for binary opening.
        closing_disk (int): Size of the disk for binary closing.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cleaned binary mask, and the grayscale image it
        was derived from.
    """
    # Look for optimal threshold
    img_gray = rgb2gray(img)
    thresh = threshold_triangle(img_gray)

    # morphology: smooth region boundaries and remove small objects
    cleaned = opening(img_gray > thresh, footprint=disk(opening_disk))
    cleaned = closing(cleaned, footprint=disk(closing_disk))
    cleaned = remove_small_objects(cleaned, max_size=min_object_size - 1)

    return cleaned, img_gray


def _select_bbox(
    img_mask: np.ndarray,
    img_intensity: np.ndarray,
    img_height: int,
    min_bbox_height: int,
    edge_margin_top: int,
    edge_margin_bottom: int,
    min_size_for_bottom: int,
) -> tuple[int, int, int, int]:
    """Select the bounding box of the core region from the list of region properties.

    Assumptions:
    - The core region is the largest region that does not touch the top edge of the image
    - The core region may touch the bottom edge of the image if it is large enough
    - The core region has a certain minimum height
    - Union of all candidate bboxes is used to handle fragmented cores

    Fallback:
    - If no candidate regions are found, the largest region is selected as the core region.

    Args:
        img_mask (np.ndarray): Binary mask of candidate core regions.
        img_intensity (np.ndarray): Grayscale intensity image used to weight region properties.
        img_height (int): Height of the input image.
        min_bbox_height (int): Minimum height for a candidate core bounding box.
        edge_margin_top (int): Ignore top edge of image (ruler).
        edge_margin_bottom (int): Ignore bottom edge of image (ruler).
        min_size_for_bottom (int): Minimum area for a candidate core to touch the bottom edge of the image.

    Returns:
        tuple[int, int, int, int]: Bounding box as (x_min, y_min, x_max, y_max), with x_max/y_max
            as inclusive coordinates.

    Raises:
        SegmentationError: If no regions are found.
    """
    props = regionprops(label(img_mask), intensity_image=img_intensity)
    if not props:
        raise SegmentationError("No regions found in image")

    candidates = [
        r
        for r in props
        if (r.bbox[2] - r.bbox[0]) > min_bbox_height  # exclude ruler
        and r.bbox[0] > edge_margin_top  # doesn't touch top edge
        and (
            r.bbox[2] <= img_height - edge_margin_bottom or r.area > min_size_for_bottom
        )  # only large regions can touch bottom
    ]
    if not candidates:
        # fallback: just pick the largest region
        candidates = [max(props, key=lambda r: r.area)]

    # union of all candidate bboxes to handle fragmented cores
    min_row = min(r.bbox[0] for r in candidates)
    min_col = min(r.bbox[1] for r in candidates)
    max_row = max(r.bbox[2] for r in candidates)
    max_col = max(r.bbox[3] for r in candidates)

    return (min_col, min_row, max_col - 1, max_row - 1)


def _find_non_tray_interval(values: np.ndarray, threshold: float, ratio: float) -> tuple[int, int]:
    """Find the largest contiguous row interval that is not tray, based on a per-pixel value map.

    A row is classified as tray if the fraction of its pixels above `threshold` reaches
    `ratio`. The largest run of consecutive non-tray rows is returned.

    Args:
        values (np.ndarray): 2D per-pixel value map (e.g. saturation channel) to threshold.
        threshold (float): Value above which a pixel is considered tray.
        ratio (float): Fraction of tray pixels in a row required to classify the row as tray.

    Returns:
        tuple[int, int]: Start and end row indices (exclusive of tray) of the largest
            non-tray interval.

    Raises:
        SegmentationError: If every row is classified as tray (no non-tray interval found),
            or if the largest non-tray interval is degenerate (zero height).
    """
    confs_row = (values > threshold).mean(axis=1)
    detections = np.nonzero(confs_row < ratio)[0]

    if detections.size == 0:
        raise SegmentationError("Entire region classified as tray; no non-tray interval found")

    groups = [[v for _, v in g] for _, g in groupby(enumerate(detections), key=lambda iv: iv[1] - iv[0])]
    result = np.array([[g[0], g[-1]] for g in groups])
    id_best = np.argmax(result[:, 1] - result[:, 0])
    top_trim, bottom_trim = result[id_best, 0].item(), result[id_best, 1].item()

    if bottom_trim <= top_trim:
        raise SegmentationError("Largest non-tray interval is degenerate (zero height)")

    return top_trim, bottom_trim


def segment_core_from_tray(
    img_metadata: ImageMetadata, bbox: ImageSegmentResult, config: SegmentationCoreConfig
) -> ImageSegmentResult:
    """Vertical trim the bounding box to exclude the wooden tray based on saturation.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and trim.
        bbox (ImageSegmentResult): Bounding box to trim, in the original image's coordinate space.
        config (SegmentationCoreConfig): Tunable segmentation parameters (uses `tray_sat_threshold`,
            `tray_sat_ratio`, and `downscale_factor`).

    Returns:
        ImageSegmentResult: Trimmed bounding box as (left, top, right, bottom), in the
            original image's coordinate space.

    Raises:
        SegmentationError: If every row is classified as tray (no non-tray interval found).
    """
    img = img_metadata.load_image(factor=config.downscale_factor)
    x_min, y_min, x_max, y_max = scale_bbox(bbox.bbox, factor=config.downscale_factor)

    hsv = rgb2hsv(img[int(y_min) : int(y_max + 1), int(x_min) : int(x_max + 1)])
    top_trim, bottom_trim = _find_non_tray_interval(hsv[:, :, 1], config.tray_sat_threshold, config.tray_sat_ratio)

    trimmed_bbox = (x_min, y_min + top_trim, x_max, y_min + bottom_trim)
    return ImageSegmentResult(bbox=scale_bbox(trimmed_bbox, factor=1 / config.downscale_factor))


def group_images_by_shape(imgs_metadata: list[ImageMetadata]) -> dict[tuple[int, int, int], list[ImageMetadata]]:
    """Group images by their shape (height, width, channels).

    Args:
        imgs_metadata (list[ImageMetadata]): List of image metadata.

    Returns:
        dict[tuple[int, int, int], list[ImageMetadata]]: Dictionary where keys are image shapes
            and values are lists of ImageMetadata with that shape.
    """
    grouped: dict[tuple[int, int, int], list[ImageMetadata]] = {}
    for img_metadata in imgs_metadata:
        grouped.setdefault(img_metadata.shape, []).append(img_metadata)
    return grouped


def segment_tray_by_group(
    imgs_metadata: list[ImageMetadata],
    config: SegmentationTrayMultipleConfig,
) -> dict[tuple[int, int, int], ImageSegmentResult]:
    """Estimate a shared tray bounding box per group of same-shaped images.

    Images are grouped by their on-disk shape (read from metadata, without decoding pixel
    data). Groups with at least ``config.n_min_foreground`` images have their foreground
    estimated from a random sample of that many images; smaller groups are skipped and left
    to the per-image fallback (segment_tray_single).

    Args:
        imgs_metadata (list[ImageMetadata]): All images to consider, potentially spanning
            multiple shapes.
        config (SegmentationTrayMultipleConfig): Tunable segmentation parameters.

    Returns:
        dict[tuple[int, int, int], ImageSegmentResult]: Estimated tray bounding box per image
            shape, for groups where estimation succeeded.
    """
    rng = np.random.default_rng(config.seed)
    groups = group_images_by_shape(imgs_metadata)
    results = {}
    for shape, group in groups.items():
        if len(group) < config.n_min_foreground:
            continue

        # subset of images is enough to calculate the foreground mask
        sample_ids = rng.choice(len(group), size=config.n_min_foreground, replace=False)
        sampled_imgs = [group[i] for i in sample_ids]
        result = segment_tray_multiple(sampled_imgs, config)
        if result is not None:
            results[shape] = result

    logger.info("Computed shared foreground for %d/%d shape group(s).", len(results), len(groups))

    return results
