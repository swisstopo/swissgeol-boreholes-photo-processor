"""Module for stitching core segments together."""

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.models import ImageMetadataProcessedCores
from src.stitching.config import CoreStitchingConfig, StitchingConfig
from src.stitching.draw import _draw_borehole_label, _draw_cores, _draw_ruler
from src.stitching.utils import _resize_images, _resize_scale

logger = logging.getLogger(__name__)


def _rounded_ruler_display_steps(shared_ruler_steps: int) -> int:
    """Round a ruler span to the nearest 50cm, for a clean, consistent look when drawn.

    This only affects how the ruler is drawn (its tick count/labels and the pixel height it's
    drawn over); it does not change shared_ruler_steps itself, which remains the actual span
    used to scale cores. A core longer than the rounded value will render past the ruler's
    last labeled tick, rather than being rescaled to fit under it.
    """
    return max(50, round(shared_ruler_steps / 50) * 50)


@dataclass
class StitchingBatchCores:
    """One chunk of cores plus the canvas-wide values needed to stitch it, for parallel dispatch."""

    cores: list[ImageMetadataProcessedCores]  # cores assigned to this batch
    shared_ruler_steps: int  # canvas-wide ruler span, shared across all batches
    shared_borehole_id: str  # borehole ID drawn in the label, shared across all batches
    fallback_scale: float  # px-per-unit fallback for cores with no detected ruler


def stitching_batch_cores(
    cores: list[ImageMetadataProcessedCores],
    shared_ruler_steps: int,
    shared_borehole_id: str,
    fallback_scale: float,
    config: StitchingConfig,
) -> Image.Image:
    """Stitch one batch (chunk) of core segments into a single output image.

    Cores are resized to a shared pixel scale (derived from each core's ruler resolution,
    or fallback_scale where no ruler was detected), pasted left to right with depth labels,
    and flanked by a depth ruler on each side of the canvas. Cores are spaced evenly so their
    combined width plus gaps exactly fills core_area_width; how many cores land in this batch
    was already decided by stitching_cores() so that every gap is at least min_core_gap.

    The values are cores_height (H), core_area_width (A), and ruler_width (R),
    padding_horizontal (PH), and padding_vertical (PV).

    <-------------------------- 4 * PH + 2 * R + A ---------------------------->
    ----------------------------------------------------------------------------  ʌ
    | ID     ʌ                                                                 |  |
    |       3*PV        FROM           FROM           FROM                     |  |
    |        v     <---------------------- A --------------------->            |  |
    | <PH> |---| <PH> |--------| ... |--------| ... |--------| <PH> |---| <PH> |  |
    |      | r |      | CORE 1 |  ʌ  | CORE J |     | CORE N |      | r |      |  |
    |      | u |      |        |  |  |        |     |        |      | u |      | 5*PV
    |      | l |      |        |  |H |        |     |        |      | l |      | + H
    |      | e |      |--------|  |  |        |     |        |      | e |      |  |
    |      | r |                  |  |        |     |------- |      | r |      |  |
    |      |---|                  v  |--------|                     |---|      |  |
    |        ʌ                                                      <-R->      |  |
    |       2*PV          TO             TO             TO                     |  |
    |        v                                                                 |  |
    ----------------------------------------------------------------------------  v

    Args:
        cores (list[ImageMetadataProcessedCores]): The list of processed image metadata objects to stitch together.
        shared_ruler_steps (int): Number of major ruler ticks (depth units) spanned by the canvas height,
            shared across all batches so rulers line up between output images.
        shared_borehole_id (str): Borehole core ID drawn in the top-left label, shared across all batches.
        fallback_scale (float): Pixels-per-unit used to resize cores whose ruler was not detected.
        config (StitchingConfig): Tunable layout parameters (padding, font size, canvas sizing, etc.).

    Returns:
        Image.Image: The stitched image for this batch of cores.
    """
    core_config = config.core
    core_area_width = core_config.core_area_width

    # Load all core crops up front so we can identify outliers before resizing
    cores_img = [core.load_core() for core in cores]

    # Resize all crops to preserve aspect ratio
    px_to_scales = [(core.ruler.px_per_unit if core.ruler else None) for core in cores]
    cores_img, valid_resizes = _resize_images(
        images=cores_img,
        scales=[
            (core_config.max_core_height / shared_ruler_steps) / (px_to_scale or fallback_scale)
            for px_to_scale in px_to_scales
        ],
        max_core_height=core_config.max_core_height,
        max_core_width=core_config.max_core_width,
    )
    # each core belongs to exactly one batch, so its full-resolution cache can be freed once
    # resized here -- otherwise every core's crop stays cached in memory for the whole run
    for core in cores:
        core.release_core_cache()

    canvas_width = (
        2 * core_config.ruler_width  # Ruler
        + 4 * core_config.padding_horizontal  # Padding
        + core_area_width  # Cores
    )
    canvas_height = 5 * core_config.padding_vertical + core_config.max_core_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

    # Drawing: spread the leftover width evenly between cores, which stitching_cores() already
    # guaranteed is at least min_core_gap by limiting how many cores it assigned to this batch
    v_core_width = sum(img.width for img in cores_img)
    n_padding_horizontal = (core_area_width - v_core_width) / max(len(cores_img) - 1, 1)
    canvas = _draw_cores(
        canvas=canvas,
        cores=cores_img,
        labels_range=[(core.depth_start, core.depth_end) for core in cores],
        is_scale_correct=[
            (px_to_scale is not None) and valid_resize
            for px_to_scale, valid_resize in zip(px_to_scales, valid_resizes, strict=True)
        ],
        loc=(2 * core_config.padding_horizontal + core_config.ruler_width, 3 * core_config.padding_vertical),
        padding_horizontal=int(n_padding_horizontal),
        padding_vertical=core_config.padding_vertical,
        max_core_height=core_config.max_core_height,
        font_size=core_config.font_size,
    )

    canvas = _draw_borehole_label(
        canvas,
        borehole_id=shared_borehole_id,
        loc=(core_config.padding_horizontal, core_config.padding_vertical),
        font_size=core_config.font_size,
    )

    # Round the ruler's displayed span to the nearest 50cm; scale its drawn pixel height to match,
    # at the same px-per-unit density used for the cores above, so its ticks stay correctly
    # positioned relative to them (a core longer than this rounded span simply extends past the
    # ruler's last labeled tick, rather than the ruler being stretched to always cover it).
    drawn_ruler_steps = _rounded_ruler_display_steps(shared_ruler_steps)
    drawn_ruler_height = round(core_config.max_core_height * drawn_ruler_steps / shared_ruler_steps)

    canvas = _draw_ruler(
        canvas,
        loc=(core_config.padding_horizontal, 3 * core_config.padding_vertical),
        size=(core_config.ruler_width, drawn_ruler_height),
        n_major=drawn_ruler_steps,
        font_size=round(core_config.font_size / 2),
    )

    canvas = _draw_ruler(
        canvas,
        loc=(
            3 * core_config.padding_horizontal + core_config.ruler_width + core_area_width,
            3 * core_config.padding_vertical,
        ),
        size=(core_config.ruler_width, drawn_ruler_height),
        n_major=drawn_ruler_steps,
        font_size=round(core_config.font_size / 2),
        horizontal_flip=True,
    )

    return canvas


def _predicted_core_width(
    core: ImageMetadataProcessedCores,
    shared_ruler_steps: int,
    fallback_scale: float,
    core_config: CoreStitchingConfig,
) -> int:
    """Estimate a core's rendered width after the resize in stitching_batch_cores(), from bbox metadata alone.

    Mirrors the scale computed for _resize_images() there, so batches can be sized by actual
    rendered width without loading any pixel data.

    Args:
        core (ImageMetadataProcessedCores): The core to estimate, with a detected bbox.
        shared_ruler_steps (int): Canvas-wide ruler span (see stitching_batch_cores()).
        fallback_scale (float): Pixels-per-unit used for cores with no detected ruler.
        core_config (CoreStitchingConfig): Tunable layout parameters (max dimensions, etc.).

    Returns:
        int: Predicted width in pixels of this core once resized for stitching.
    """
    width, height = core.core_dimensions
    px_to_scale = core.ruler.px_per_unit if core.ruler else None
    scale = (core_config.max_core_height / shared_ruler_steps) / (px_to_scale or fallback_scale)
    scale = _resize_scale(width, height, scale, core_config.max_core_width, core_config.max_core_height)
    return round(width * scale)


def _chunk_cores_by_width(
    imgs: list[ImageMetadataProcessedCores],
    predicted_widths: list[int],
    core_area_width: int,
    min_core_gap: int,
) -> list[list[ImageMetadataProcessedCores]]:
    """Greedily group cores left to right so each page's cores plus minimum gaps fit core_area_width.

    A core that alone exceeds core_area_width still gets a page of its own, rather than being dropped.

    Args:
        imgs (list[ImageMetadataProcessedCores]): Cores in display order.
        predicted_widths (list[int]): Predicted rendered width per core, same order as imgs.
        core_area_width (int): Fixed pixel budget for cores on one page.
        min_core_gap (int): Minimum pixel gap to reserve between adjacent cores when packing.

    Returns:
        list[list[ImageMetadataProcessedCores]]: One list of cores per page.
    """
    chunks: list[list[ImageMetadataProcessedCores]] = []
    current: list[ImageMetadataProcessedCores] = []
    current_width = 0

    for img, width in zip(imgs, predicted_widths, strict=True):
        gap = min_core_gap if current else 0
        if current and current_width + gap + width > core_area_width:
            chunks.append(current)
            current, current_width, gap = [], 0, 0
        current.append(img)
        current_width += gap + width

    if current:
        chunks.append(current)

    return chunks


def stitching_cores(
    imgs: list[ImageMetadataProcessedCores],
    config: StitchingConfig,
) -> list[StitchingBatchCores]:
    """Split cores into chunks and compute the canvas-wide values shared across all of them.

    Cores are assigned to pages greedily by their predicted rendered width, so that each page's
    cores plus at least min_core_gap between them fit within core_area_width. This decides how
    many cores land on a page; the actual (larger-or-equal) gap is only computed once stitching
    the page, from the cores' true resized widths (see stitching_batch_cores()).

    Args:
        imgs (list[ImageMetadataProcessedCores]): The list of processed image metadata objects to stitch together.
        config (StitchingConfig): Tunable layout parameters (padding, font size, canvas sizing, etc.).

    Returns:
        list[StitchingBatchCores]: One batch per page of cores.
    """
    # Get spans and resolution for all cores
    original = np.array(
        [
            (img.ruler.px_per_unit, (img.core.bbox[2] - img.core.bbox[0]))
            for img in imgs
            # Both ruler and core need to be detected for scaling to work
            if img.ruler and img.core
        ]
    ).T

    if original.size == 0:
        logger.warning("No detection has both a ruler and a core; nothing to stitch")
        return []

    original_scales, original_heights = original

    # Set default resolution if missing
    fallback_scale = np.median(original_scales).item()

    # Estimate ruler span over all cores
    canvas_ruler_steps = np.ceil(max(original_heights / original_scales)).astype(int).item()

    predicted_widths = [_predicted_core_width(img, canvas_ruler_steps, fallback_scale, config.core) for img in imgs]
    chunks = _chunk_cores_by_width(imgs, predicted_widths, config.core.core_area_width, config.core.min_core_gap)

    return [
        StitchingBatchCores(
            cores=chunk,
            shared_ruler_steps=canvas_ruler_steps,
            shared_borehole_id=imgs[0].borehole_id,
            fallback_scale=fallback_scale,
        )
        for chunk in chunks
    ]
