"""Tray/core bbox detection: per-image fallback and shared-foreground estimation across a batch."""

import logging
from timeit import default_timer as timer

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from sklearn.mixture import GaussianMixture

from src.config import (
    SegmentationError,
    SegmentationTrayGroupConfig,
    SegmentationTraySingleConfig,
)
from src.models import (
    ImageMetadata,
    RulerSegmentResult,
    TraySegmentResult,
)
from src.segment.utils.misc import ProcessGroupByShape
from src.utils import scale_bbox

logger = logging.getLogger(__name__)


def bbox_skimage_interserction(
    bboxA: tuple[float, float, float, float], bboxB: tuple[float, float, float, float]
) -> bool:
    """Check whether two skimage-style bounding boxes overlap.

    Args:
        bboxA (tuple[float, float, float, float]): First bounding box in skimage
            regionprops format (min_row, min_col, max_row, max_col).
        bboxB (tuple[float, float, float, float]): Second bounding box in skimage
            regionprops format (min_row, min_col, max_row, max_col).

    Returns:
        bool: True if the two bounding boxes overlap, False otherwise.
    """
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    return abs(max((xB - xA, 0)) * max((yB - yA), 0)) != 0


def _apply_threshold_and_clean(
    img: np.ndarray,
    min_object_size: int,
    opening_disk: int,
    closing_disk: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholding to the input image and return a cleaned binary mask.

    Args:
        img (np.ndarray): RGB image array (float, [0, 1]) to be thresholded.
        min_object_size (int): Minimum size of objects to be retained.
        opening_disk (int): Size of the disk for binary opening.
        closing_disk (int): Size of the disk for binary closing.
        block_size (int): Pixel neighborhood used to calculate the threshold.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cleaned binary mask, and the grayscale image it
        was derived from.
    """
    # Look for optimal threshold
    img_gray = rgb2gray(img)
    local_thresh = threshold_local(img_gray, block_size=(2 * (block_size // 2) + 1))

    # morphology: smooth region boundaries and remove small objects
    cleaned = opening(img_gray > local_thresh, footprint=disk(opening_disk))
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
    exclude_area: tuple[float, float, float, float] | None,
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
        exclude_area (tuple[float, float, float, float] | None): TODO.

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

    # If exclusion area, remove bbox overlapping
    if exclude_area is not None:
        bbox_skimage_exclude = (int(exclude_area[1]), int(exclude_area[0]), int(exclude_area[3]), int(exclude_area[2]))
        candidates = [
            candidate
            for candidate in candidates
            if not bbox_skimage_interserction(bbox_skimage_exclude, candidate.bbox)
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


def segment_tray(
    img_metadata: ImageMetadata,
    config: SegmentationTraySingleConfig,
    ruler: RulerSegmentResult | None = None,
) -> TraySegmentResult:
    """Segment a single image via thresholding when no shared foreground bbox is available.

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and segment.
        config (SegmentationTraySingleConfig): Tunable segmentation parameters.
        ruler (RulerSegmentResult | None): TODO.

    Returns:
        TraySegmentResult: Bounding box as (x_min, y_min, x_max, y_max), in the original image's
            coordinate space. Background/foreground debug images are left unset.
    """
    t_start = timer()
    factor = config.downscale_factor
    binary, grey = _apply_threshold_and_clean(
        img_metadata.load_image(factor=factor),
        min_object_size=max(1, round(config.min_object_size * factor**2)),  # factor**2 for area-based configs
        opening_disk=max(1, round(config.opening_disk * factor)),
        closing_disk=max(1, round(config.closing_disk * factor)),
        block_size=max(1, round(config.block_size * factor)),
    )

    bbox = _select_bbox(
        img_mask=binary,
        img_intensity=grey,
        img_height=binary.shape[0],
        min_bbox_height=max(1, round(config.min_bbox_height * factor)),
        edge_margin_top=round(config.edge_margin_top * factor),
        edge_margin_bottom=round(config.edge_margin_bottom * factor),
        min_size_for_bottom=round(config.min_size_for_bottom * factor**2),
        exclude_area=scale_bbox(ruler.bbox, config.downscale_factor) if ruler is not None else None,
    )

    return TraySegmentResult(
        bbox=scale_bbox(bbox, factor=1 / factor),
        time=timer() - t_start,
    )


class ProcessTrayGroupByShape(ProcessGroupByShape[TraySegmentResult, np.ndarray]):
    """Estimate a shared tray/core bounding box for a group of same-shaped images."""

    def __init__(
        self,
        config: SegmentationTrayGroupConfig,
        n_workers: int = 1,
    ):
        """Configure the tray group segmentation.

        Args:
            config (SegmentationTrayGroupConfig): Tunable segmentation parameters.
            n_workers (int): Number of worker processes used to run `_preprocess` in parallel.
        """
        super().__init__(min_group_size=config.n_min_foreground, seed=config.seed, n_workers=n_workers)
        self.config = config

    def _preprocess(self, img_metadata: ImageMetadata) -> np.ndarray | None:
        """Load and blur a single image for foreground/background std estimation.

        Args:
            img_metadata (ImageMetadata): Metadata of the image to load.

        Returns:
            np.ndarray | None: Blurred grayscale image array, or None.
        """
        try:
            img = img_metadata.load_image(factor=self.config.downscale_factor)
        except (SegmentationError, ValueError) as e:
            logger.warning("%s. Skipping.", e)
            return None

        img_gray_ = rgb2gray(img)
        img_blur_ = gaussian(img_gray_, sigma=self.config.foreground_blur_sigma)
        return img_blur_

    def _aggregate(self, processed_items: list[np.ndarray]) -> TraySegmentResult | None:
        """Estimate a shared tray/core bounding box from the group's blurred grayscale images.

        Args:
            processed_items (list[np.ndarray]): Blurred grayscale images for one shape group.

        Returns:
            TraySegmentResult | None: Result with the estimated bbox and debug background/
                foreground images, or None.
        """
        unique_shapes = set([item.shape for item in processed_items])

        # At least n_min_foreground images, and all of consistent shape
        if len(processed_items) < self.config.n_min_foreground or len(unique_shapes) != 1:
            return None

        # Compute STD between images to highlight changes in background
        bg_imgs = np.stack(processed_items)
        fg_img = bg_imgs.std(axis=0)
        fg_bbox = _estimate_tray_bbox(fg_img)
        if fg_bbox is None:
            return None

        return TraySegmentResult(
            bbox=scale_bbox(fg_bbox, factor=1 / self.config.downscale_factor),
            img_background=bg_imgs.mean(axis=0),
            img_foreground=fg_img,
            img_downscale_factor=self.config.downscale_factor,
        )
