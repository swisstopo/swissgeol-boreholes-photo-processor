"""Low-level helpers for pebble cuttings detection: paper-sheet edge checks and stripe confirmation."""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk

from src.segment.config import SegmentationCuttingsPebbleConfig


def touches_bottom(p, h: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the bottom edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        h (int): Height of the image, in pixels.
        edge_margin (float): Fraction of the image height the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the bottom edge.
    """
    return p.bbox[2] >= (1 - edge_margin) * h


def touches_right(p, w: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the right edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        w (int): Width of the image, in pixels.
        edge_margin (float): Fraction of the image width the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the right edge.
    """
    return p.bbox[3] >= (1 - edge_margin) * w


def touches_left(p, w: int, edge_margin: float) -> bool:
    """Check whether a region's bounding box reaches the left edge of the image.

    Args:
        p: A skimage regionprops region, whose bbox is (min_row, min_col, max_row, max_col).
        w (int): Width of the image, in pixels.
        edge_margin (float): Fraction of the image width the region must reach to count as edge-anchored.

    Returns:
        bool: True if the region's bbox reaches within `edge_margin` of the left edge.
    """
    return p.bbox[1] <= edge_margin * w


def touches_top(p, h: int, edge_margin: float) -> bool:
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
):
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


def has_stripe_pattern(v_patch: np.ndarray, min_drop: float, max_width_frac: float) -> bool:
    """Check whether a brightness patch carries the paper card's printed depth-marker ticks.

    Every real paper card carries printed depth-marker ticks: 1-4 narrow, sharply darker
    vertical bands against an otherwise uniform background. A pebble cluster mistaken for
    paper never has this.

    Args:
        v_patch (np.ndarray): Brightness (HSV value channel) patch to search for stripes.
        min_drop (float): Brightness drop, relative to the patch's 85th percentile, that counts as a dark stripe.
        max_width_frac (float): Stripes wider than this fraction (x3) of the patch are treated as a shadow, not a tick.

    Returns:
        bool: True if at least one narrow dark stripe is found.
    """
    col = v_patch.mean(axis=0)
    baseline = np.percentile(col, 85)
    dark = col < (baseline - min_drop)
    n_stripes = 0
    i = 0
    while i < len(dark):
        if dark[i]:
            j = i
            while j < len(dark) and dark[j]:
                j += 1
            if j - i <= max_width_frac * len(dark) * 3:  # narrow-ish, not a broad shadow
                n_stripes += 1
            i = j
        else:
            i += 1
    return n_stripes >= 1


def confirm_stripe(p, hsv: np.ndarray, h: int, w: int, config: SegmentationCuttingsPebbleConfig):
    """Reject a paper candidate outright if no stripe pattern is found around it.

    The shape filter in `detect_paper` already tends to exclude the striped part of the card
    itself, so the stripes are looked for just outside the candidate's own bbox, not inside it.
    We'd rather not crop at all than crop something that was never actually the paper.

    Args:
        p: A skimage regionprops region (the paper candidate), or None.
        hsv (np.ndarray): HSV image the candidate was detected in.
        h (int): Height of the image, in pixels.
        w (int): Width of the image, in pixels.
        config (SegmentationCuttingsPebbleConfig): Tunable segmentation parameters.

    Returns:
        The input region `p` if a stripe pattern is confirmed around it, otherwise None.
    """
    if p is None:
        return None
    minr, minc, maxr, maxc = p.bbox
    ph, pw = maxr - minr, maxc - minc
    r0, r1 = max(0, minr - ph), min(h, maxr + ph)
    c0, c1 = max(0, minc - pw), min(w, maxc + pw)
    patch = hsv[r0:r1, c0:c1, 2]
    return p if has_stripe_pattern(patch, config.stripe_min_drop, config.stripe_max_width_frac) else None
