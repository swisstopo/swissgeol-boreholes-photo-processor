"""Low-level helpers for pebble cuttings detection.

Paper-sheet region scoring, edge checks, and per-shape-group paper-position estimation
across a batch.
"""

import logging
from typing import Any

import numpy as np
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening

from src.config import SegmentationError
from src.models import CuttingsSegmentResult, ImageMetadataCuttings, PaperDetectionStatus
from src.segment.config import SegmentationCuttingsPebbleConfig, SegmentationCuttingsPebbleGroupConfig
from src.segment.utils.misc import ProcessGroupByShape
from src.utils import scale_bbox

logger = logging.getLogger(__name__)

# skimage.measure.regionprops returns RegionProperties, but that class lives in a private
# module (skimage.measure._regionprops); alias to Any rather than import it.
RegionProps = Any


def touches_bottom(p: RegionProps, h: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the bottom edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        h (int): Height of the image, in pixels.
        edge_margin (float): Fraction of the image height the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the bottom edge.
    """
    return p.bbox[2] >= (1 - edge_margin) * h


def touches_right(p: RegionProps, w: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the right edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        w (int): Width of the image, in pixels.
        edge_margin (float): Fraction of the image width the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the right edge.
    """
    return p.bbox[3] >= (1 - edge_margin) * w


def touches_left(p: RegionProps, w: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the left edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        w (int): Width of the image, in pixels.
        edge_margin (float): Fraction of the image width the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the left edge.
    """
    return p.bbox[1] <= edge_margin * w


def touches_top(p: RegionProps, h: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the top edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        h (int): Height of the image, in pixels.
        edge_margin (float): Fraction of the image height the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the top edge.
    """
    return p.bbox[0] <= edge_margin * h


def detect_paper(
    hsv: np.ndarray,
    h: int,
    w: int,
    v_thresh: float,
    downscale_factor: float,
    config: SegmentationCuttingsPebbleConfig,
) -> RegionProps | None:
    """Detect the reference paper sheet as the best-scoring bright, colorless region.

    The paper is a solid rectangle -- it should fill nearly all of its own bounding box
    (extent) and be close to convex (solidity). A merged blob of stones is porous (lots of
    black gaps between pebbles survive closing) and scores much lower on both, even when its
    raw area is larger. The paper is always anchored to an edge of the *original* photo's
    bottom, but load_image rotates portrait photos 90 degrees to landscape, which moves that
    edge to the right rather than the bottom -- so a candidate is only trustworthy if it
    reaches the bottom OR the right edge, no matter how high its extent/solidity. It's also
    always a modest slice of the frame; a blob covering most of the image (e.g. a wash of pale
    sand/rock filling most of the shot, which can coincidentally touch both edges) is never the
    paper regardless of shape score, and a tiny noise speck (a few stray bright pixels) can
    trivially score a perfect shape too, so a minimum area rules those out as well.

    Args:
        hsv (np.ndarray): HSV image to search for the paper candidate.
        h (int): Height of the image, in pixels.
        w (int): Width of the image, in pixels.
        v_thresh (float): Brightness (HSV value channel) threshold above which a pixel counts as paper.
        downscale_factor (float): Factor the image was downscaled by, used to scale the closing disk radius.
        config (SegmentationCuttingsPebbleConfig): Tunable segmentation parameters.

    Returns:
        The largest surviving region (a skimage regionprops object), or None if no candidate passes.
    """
    raw_mask = (hsv[..., 2] > v_thresh) & (hsv[..., 1] < config.sat_threshold)  # bright AND colorless
    # the black stripes punch holes in the paper — close them
    mask = closing(raw_mask, disk(max(1, round(config.closing_disk * downscale_factor))))
    props = regionprops(label(mask))
    candidates = [
        p
        for p in props
        if p.extent >= config.min_extent
        and p.solidity >= config.min_solidity
        and config.min_area_frac * h * w <= p.area <= config.max_area_frac * h * w
        and (touches_bottom(p, h, config.edge_margin) or touches_right(p, w, config.edge_margin))
        and not (
            touches_top(p, h, config.edge_margin) and touches_left(p, w, config.edge_margin)
        )  # ignore degenerate candidates touching both top and left
    ]
    return max(candidates, key=lambda p: p.area) if candidates else None


def resolve_paper_crop(
    paper: RegionProps | None,
    h: int,
    w: int,
    max_cropped_frac: float,
) -> tuple[PaperDetectionStatus, tuple[float, float, float, float]]:
    """Turn a detected paper candidate (or lack thereof) into a crop status and bbox.

    The paper always sits toward the bottom-right of the frame, so the cuttings region is
    always everything to the left of its left edge -- never crop by height. A degenerate paper
    region (left edge at column 0, i.e. nothing left to keep) usually just means the reference
    sheet isn't reliably in frame, so the whole image is already the cuttings region. The paper
    is also always a modest slice of the frame, so a candidate that would crop away more than
    half the image is more likely a misdetection than a real card.

    Shared between per-image (segment_pebble) and per-shape-group (ProcessPebblePaperGroupByShape)
    paper detection, which differ only in how `paper` was found.

    Args:
        paper: Detected paper region (as returned by detect_paper/detect_group_paper), or None.
        h (int): Height of the image the candidate was detected in, in pixels.
        w (int): Width of the image the candidate was detected in, in pixels.
        max_cropped_frac (float): A paper candidate cropping away more than this fraction of the
            image is rejected.

    Returns:
        tuple[PaperDetectionStatus, tuple[float, float, float, float]]: The detection outcome,
        and the resulting cuttings bbox (0, 0, right, h) in the same coordinate space as `paper`.
    """
    if paper is None:
        return PaperDetectionStatus.NO_CANDIDATE, (0, 0, w, h)
    if paper.bbox[1] == 0:
        return PaperDetectionStatus.DEGENERATE_LEFT_EDGE, (0, 0, w, h)
    if (w - paper.bbox[1]) > max_cropped_frac * w:
        return PaperDetectionStatus.CROPPED_TOO_MUCH, (0, 0, w, h)
    return PaperDetectionStatus.FOUND, (0, 0, paper.bbox[1], h)


def detect_group_paper(
    mean_img: np.ndarray,
    std_map: np.ndarray,
    downscale_factor: float,
    config: SegmentationCuttingsPebbleGroupConfig,
) -> RegionProps | None:
    """Detect the paper sheet shared by a group of same-shaped images, from their pixel statistics.

    Per-image brightness thresholding (detect_paper) breaks down when run on a group's mean
    image: averaging many different rock fragments washes the background toward a uniform gray,
    which is nearly as bright/colorless as the paper and swamps it under a single threshold. But
    the paper is the one region that stays sharp across the average -- it sits at the same pixel
    position in every shot, while the rock content differs shot to shot and blurs away -- so
    cross-image standard deviation isolates it far more cleanly than brightness alone. Brightness/
    colorlessness on the mean image is layered on top only to reject other things that are *also*
    consistent across every shot but aren't paper, such as a dark lens-vignette corner.

    Args:
        mean_img (np.ndarray): Per-pixel mean across the group's images, RGB float in [0, 255].
        std_map (np.ndarray): Per-pixel standard deviation across the group's images, averaged
            over the RGB channels.
        downscale_factor (float): Factor the group's images were downscaled by, used to scale
            the closing disk radius.
        config (SegmentationCuttingsPebbleGroupConfig): Tunable segmentation parameters.

    Returns:
        The largest surviving region (a skimage regionprops object), or None if no candidate passes.
    """
    h, w = std_map.shape
    hsv = rgb2hsv(mean_img / 255.0)
    std_thresh = np.percentile(std_map, config.std_percentile)
    raw_mask = (std_map < std_thresh) & (hsv[..., 2] > config.val_threshold) & (hsv[..., 1] < config.sat_threshold)
    # the black stripes punch holes in the paper — close them; opening then drops small noise
    # blobs that pass the std/brightness thresholds by chance (more likely with smaller groups)
    mask = closing(raw_mask, disk(max(1, round(config.closing_disk * downscale_factor))))
    mask = opening(mask, disk(2))
    props = regionprops(label(mask))
    candidates = [
        p
        for p in props
        if p.extent >= config.min_extent
        and p.solidity >= config.min_solidity
        and config.min_area_frac * h * w <= p.area <= config.max_area_frac * h * w
        and (touches_bottom(p, h, config.edge_margin) or touches_right(p, w, config.edge_margin))
        and not (
            touches_top(p, h, config.edge_margin) and touches_left(p, w, config.edge_margin)
        )  # ignore degenerate candidates touching both top and left
    ]
    return max(candidates, key=lambda p: p.area) if candidates else None


class ProcessPebblePaperGroupByShape(ProcessGroupByShape[ImageMetadataCuttings, CuttingsSegmentResult, np.ndarray]):
    """Estimate a shared pebble paper-sheet crop for a group of same-shaped images."""

    def __init__(
        self,
        config: SegmentationCuttingsPebbleGroupConfig,
        n_workers: int = 1,
    ):
        """Configure the pebble paper group estimation.

        Args:
            config (SegmentationCuttingsPebbleGroupConfig): Tunable segmentation parameters.
            n_workers (int): Number of worker processes used to run `_preprocess` in parallel.
        """
        super().__init__(min_group_size=config.n_min_group, seed=config.seed, n_workers=n_workers)
        self.config = config

    def _preprocess(
        self, img_metadata: ImageMetadataCuttings, img_metadata_ref: ImageMetadataCuttings
    ) -> np.ndarray | None:
        """Load and downscale a single image ahead of stacking; no alignment to the reference is needed.

        Args:
            img_metadata (ImageMetadataCuttings): Metadata of the image to load.
            img_metadata_ref (ImageMetadataCuttings): Unused; images are stacked as-is since the
                camera framing (not the cuttings) is assumed consistent within a shape group.

        Returns:
            np.ndarray | None: RGB image array, float in [0, 255], or None if the image failed to load.
        """
        try:
            img = img_metadata.load_image(factor=self.config.downscale_factor)
        except (SegmentationError, ValueError) as e:
            logger.warning(f"Skipping. {e}.")
            return None
        return img * 255.0

    def _aggregate(self, processed_items: list[np.ndarray]) -> CuttingsSegmentResult | None:
        """Estimate the shared paper crop from the group's cross-image mean and standard deviation.

        Args:
            processed_items (list[np.ndarray]): Downscaled RGB images for one shape group.

        Returns:
            CuttingsSegmentResult | None: Result with the estimated crop bbox and paper detection
                status, or None if the group is too small or inconsistently shaped.
        """
        unique_shapes = {item.shape for item in processed_items}

        # At least n_min_group images, and all of consistent shape
        if len(processed_items) < self.config.n_min_group or len(unique_shapes) != 1:
            return None

        stack = np.stack(processed_items, axis=0)
        mean_img = stack.mean(axis=0)
        std_map = stack.std(axis=0).mean(axis=-1)
        h, w = std_map.shape

        paper = detect_group_paper(mean_img, std_map, self.config.downscale_factor, self.config)
        status, bbox = resolve_paper_crop(paper, h, w, self.config.max_cropped_frac)

        return CuttingsSegmentResult(
            bbox=scale_bbox(bbox, factor=1 / self.config.downscale_factor),
            paper_status=status,
        )
