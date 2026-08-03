"""Module for stitching core segments together."""

import logging
from collections.abc import Generator

import numpy as np
from PIL import Image

from src.models import ImageMetadataProcessed
from src.stitching.config import StitchingConfig
from src.stitching.draw import _draw_borehole_label, _draw_cores, _draw_ruler
from src.stitching.utils import _resize_images

logger = logging.getLogger(__name__)


def stitching_batch(
    cores: list[ImageMetadataProcessed],
    shared_ruler_steps: int,
    shared_core_id: str,
    fallback_scale: float,
    config: StitchingConfig,
) -> Image.Image:
    """Stitch one batch (chunk) of core segments into a single output image.

    Cores are resized to a shared pixel scale (derived from each core's ruler resolution,
    or fallback_scale where no ruler was detected), pasted left to right with depth labels,
    and flanked by a depth ruler on each side of the canvas.

    The values are cores_height (H), cores_width (W), and ruler_width (R),
    padding_horizontal (PH), padding_vertical (PV), and num_cores_per_image (N).

    <------------------------ 4 * PH + 2 * R + W * N -------------------------->
    ----------------------------------------------------------------------------  ʌ
    | ID     ʌ                                                                 |  |
    |       3*PV        FROM           FROM           FROM                     |  |
    |        v     <------------------ W * N ------------------>               |  |
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
        cores (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        shared_ruler_steps (int): Number of major ruler ticks (depth units) spanned by the canvas height,
            shared across all batches so rulers line up between output images.
        shared_core_id (str): Borehole core ID drawn in the top-left label, shared across all batches.
        fallback_scale (float): Pixels-per-unit used to resize cores whose ruler was not detected.
        config (StitchingConfig): Tunable layout parameters (padding, font size, canvas sizing, etc.).

    Returns:
        Image.Image: The stitched image for this batch of cores.
    """
    core_config = config.core
    n_core_width = core_config.num_cores_per_image * core_config.max_core_width

    # Load all core crops up front so we can identify outliers before resizing
    cores_img = [core.load_core() for core in cores]

    # resize all crops to preserve aspect ratio
    cores_img = _resize_images(
        images=cores_img,
        scales=[
            (core_config.max_core_height / shared_ruler_steps)
            / ((s.ruler.px_per_unit if s.ruler else None) or fallback_scale)
            for s in cores
        ],
        max_core_height=core_config.max_core_height,
        max_core_width=core_config.max_core_width,
    )

    canvas_width = (
        2 * core_config.ruler_width  # Ruler
        + 4 * core_config.padding_horizontal  # Padding
        + n_core_width  # Cores
    )
    canvas_height = 5 * core_config.padding_vertical + core_config.max_core_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

    # Drawing
    v_core_width = sum([img.width for img in cores_img])  # Width of all core
    v_core_width = v_core_width * (core_config.num_cores_per_image / len(cores_img))  # Add width if core is missing
    n_padding_horizontal = (n_core_width - v_core_width) / max((core_config.num_cores_per_image - 1), 1)
    canvas = _draw_cores(
        canvas=canvas,
        cores=cores_img,
        labels_range=[(core.depth_start, core.depth_end) for core in cores],
        loc=(2 * core_config.padding_horizontal + core_config.ruler_width, 3 * core_config.padding_vertical),
        padding_horizontal=int(n_padding_horizontal),
        padding_vertical=core_config.padding_vertical,
        max_core_height=core_config.max_core_height,
        font_size=core_config.font_size,
    )

    canvas = _draw_borehole_label(
        canvas,
        borehole_id=shared_core_id,
        loc=(core_config.padding_horizontal, core_config.padding_vertical),
        font_size=core_config.font_size,
    )

    canvas = _draw_ruler(
        canvas,
        loc=(core_config.padding_horizontal, 3 * core_config.padding_vertical),
        size=(core_config.ruler_width, core_config.max_core_height),
        n_major=shared_ruler_steps,
        font_size=round(core_config.font_size / 2),
    )

    canvas = _draw_ruler(
        canvas,
        loc=(
            3 * core_config.padding_horizontal + core_config.ruler_width + n_core_width,
            3 * core_config.padding_vertical,
        ),
        size=(core_config.ruler_width, core_config.max_core_height),
        n_major=shared_ruler_steps,
        font_size=round(core_config.font_size / 2),
        horizontal_flip=True,
    )

    return canvas


def stitching(
    imgs: list[ImageMetadataProcessed],
    config: StitchingConfig,
) -> Generator[Image.Image, None, None]:
    """Stitch core segments together, yielding one output image at a time.

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        config (StitchingConfig): Tunable layout parameters (padding, font size, canvas sizing, etc.).

    Yields:
        Image.Image: One stitched image per chunk of up to num_cores_per_image cores.
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
        return

    original_scales, original_heights = original

    # Set default resolution if missing
    fallback_scale = np.median(original_scales).item()

    # Estimate ruler span over all cores
    canvas_ruler_steps = np.ceil(max(original_heights / original_scales)).astype(int).item()

    for i in range(0, len(imgs), config.core.num_cores_per_image):
        yield stitching_batch(
            cores=imgs[i : i + config.core.num_cores_per_image],
            shared_ruler_steps=canvas_ruler_steps,
            shared_core_id=imgs[0].borehole_id,
            fallback_scale=fallback_scale,
            config=config,
        )
