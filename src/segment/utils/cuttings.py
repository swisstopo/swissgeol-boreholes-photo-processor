"""Low-level helpers for pebble cuttings detection: paper-sheet region scoring and edge checks."""

from typing import Any

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk

from src.segment.config import SegmentationCuttingsPebbleConfig

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
