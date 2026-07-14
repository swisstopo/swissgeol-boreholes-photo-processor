"""Module for stitching core segments together."""

from collections.abc import Generator

from PIL import Image

from src.models import ImageMetadataProcessed
from src.stitching.draw import _draw_borehole_label, _draw_cores, _draw_ruler
from src.stitching.utils import _resize_cores


def stitching_batch(
    cores: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    padding_vertical: int = 200,
    padding_horizontal: int = 150,
    ruler_width: int = 300,
    core_height_px: int = 10000,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
    font_size: int = 100,
) -> Image.Image:
    """Stitch one batch (chunk) of core segments into a single output image.

    Each core crop is resized preserving its aspect ratio, scaling its own pixel
    height to match core_height_px * core_height_m (see _resize_cores). Cores whose
    own scale factor would make them disproportionately wide relative to a reference
    core in the same chunk are instead rescaled using the reference core's scale
    factor, keeping pixel density consistent across all cores.

    Note: this scaling is currently based on each crop's raw pixel size, not on its
    depth interval (depth_end - depth_start) — core height is not yet depth-accurate.

    The values are padding_horizontal (A), padding_vertical (B),
    core_height_px (C), and ruler_width (D), and num_cores_per_image (N).

    <-------------- (3 + N) * A + 2 * D + sum (core widths) --------------->
    ------------------------------------------------------------------------  ʌ
    | ID    ʌ                                                              |  |
    |      3*B         FROM           FROM           FROM                  |  |
    |       v                                                              |  |
    | <A> |---| <A> |--------| ... |--------| ... |--------| <A> |---| <A> |  |
    |     | r |     | CORE 1 |  ʌ  | CORE J |     | CORE N |     | r |     |  |
    |     | u |     |        |  |  |        |     |        |     | u |     | 5*B
    |     | l |     |        |  |C |        |     |        |     | l |     | + C
    |     | e |     |--------|  |  |        |     |        |     | e |     |  |
    |     | r |                 |  |        |     |------- |     | r |     |  |
    |     |---|                 v  |--------|                    |---|     |  |
    |       ʌ                                                    <-D->     |  |
    |      2*B          TO             TO             TO                   |  |
    |       v                                                              |  |
    ------------------------------------------------------------------------  v

    Args:
        cores (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): Expected number of cores on image. Pad with mean width if not enough cores.
        padding_vertical (int): Top/bottom border height in pixels.
        padding_horizontal (int): Left/right border width in pixels.
        ruler_width (int): Width in pixels of each of the two depth rulers.
        core_height_px (int): Pixel budget for a core spanning core_height_m metres.
        core_height_m (float): Depth extent, in metres, represented by core_height_px pixels.
        core_width_rerror (float): Maximum allowed width ratio vs. the reference core before a core is
            treated as an outlier.
        font_size (int): Font size used for labels.

    Returns:
        Image.Image: The stitched image for this batch of cores.
    """
    # Load all core crops up front so we can identify outliers before resizing
    cores_img = [core.load_core() for core in cores]

    # resize all crops to preserve aspect ratio
    cores_img = _resize_cores(
        cores=cores_img,
        core_height_px=core_height_px,
        core_height_m=core_height_m,
        core_width_rerror=core_width_rerror,
    )

    # Derive gap from remaining horizontal space after placing all cores
    cores_widths = [core_img.width for core_img in cores_img]

    # Pad widths if not enough cores on image
    if len(cores) < num_cores_per_image:
        cores_widths = cores_widths + [int(sum(cores_widths) / len(cores))] * (num_cores_per_image - len(cores))

    canvas_width = (
        (3 + num_cores_per_image) * padding_horizontal  # Paddings
        + 2 * ruler_width  # Ruler
        + sum(cores_widths)  # Cores
    )
    canvas_height = 5 * padding_vertical + core_height_px
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

    # Drawing
    canvas = _draw_cores(
        canvas=canvas,
        cores=cores_img,
        labels_range=[(core.depth_start, core.depth_end) for core in cores],
        loc=(2 * padding_horizontal + ruler_width, 3 * padding_vertical),
        padding_horizontal=padding_horizontal,
        padding_vertical=padding_vertical,
        font_size=font_size,
    )

    canvas = _draw_borehole_label(
        canvas,
        borehole_id=cores[0].borehole_id,
        loc=(padding_horizontal, padding_vertical),
        font_size=font_size,
    )

    canvas = _draw_ruler(
        canvas,
        loc=(padding_horizontal, 3 * padding_vertical),
        size=(ruler_width, core_height_px),
        n_major=100,
        font_size=round(font_size / 2),
    )

    canvas = _draw_ruler(
        canvas,
        loc=((2 + num_cores_per_image) * padding_horizontal + ruler_width + sum(cores_widths), 3 * padding_vertical),
        size=(ruler_width, core_height_px),
        n_major=100,
        font_size=round(font_size / 2),
        horizontal_flip=True,
    )

    return canvas


def stitching(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    padding_vertical: int = 200,
    padding_horizontal: int = 150,
    ruler_width: int = 300,
    core_height_px: int = 10000,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
    font_size: int = 100,
) -> Generator[Image.Image, None, None]:
    """Stitch core segments together, yielding one output image at a time.

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): Number of cores placed side by side per output sheet.
        padding_vertical (int): Top/bottom border height in pixels.
        padding_horizontal (int): Left/right border width in pixels.
        ruler_width (int): Width in pixels of each of the two depth rulers.
        core_height_px (int): Pixel budget for a core spanning core_height_m metres.
        core_height_m (float): Depth extent, in metres, represented by core_height_px pixels.
        core_width_rerror (float): Maximum allowed width ratio vs. the reference core before a core is
            treated as an outlier.
        font_size (int): Font size used for labels.

    Yields:
        Image.Image: One stitched image per chunk of up to num_cores_per_image cores.
    """
    for i in range(0, len(imgs), num_cores_per_image):
        yield stitching_batch(
            cores=imgs[i : i + num_cores_per_image],
            num_cores_per_image=num_cores_per_image,
            padding_vertical=padding_vertical,
            padding_horizontal=padding_horizontal,
            ruler_width=ruler_width,
            core_height_px=core_height_px,
            core_height_m=core_height_m,
            core_width_rerror=core_width_rerror,
            font_size=font_size,
        )
