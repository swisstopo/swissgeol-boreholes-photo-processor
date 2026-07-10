"""Module for stitching core segments together."""

from collections.abc import Generator

import numpy as np
from PIL import Image

from src.models import CoreSegmentResult, ImageMetadataProcessed
from src.stitching.draw import _content_x_start, _draw_borehole_label, _draw_cores, _draw_ruler


def _cut_core(source: Image.Image, result: CoreSegmentResult) -> Image.Image:
    """Cut a core segment from the source image, rotating to portrait if needed.

    Cores are stored vertically in the output, so landscape crops (width > height)
    are rotated 90° clockwise so the left edge (shallow end) becomes the top.

    Args:
        source (Image.Image): The source image from which to cut the core segment.
        result (CoreSegmentResult): The result containing the bounding box for the core segment.

    Returns:
        Image.Image: The cropped core segment image in portrait orientation.
    """
    src = source.copy()
    left, upper, right, lower = (round(v) for v in result.bounding_box)
    crop = src.crop((left, upper, right, lower))
    if crop.width > crop.height:
        crop = crop.transpose(Image.Transpose.ROTATE_270)  # clockwise: left (shallow) → top
    return crop


def _resize_core(
    crop: Image.Image,
    depth_start: float,
    depth_end: float,
    core_strip_height: int,
    max_core_length_m: float = 1.0,
) -> Image.Image:
    """Resize a core crop so its height is proportional to its depth extent.

    The aspect ratio of the original crop is preserved — only the height is
    derived from the depth interval, and the width scales accordingly.  This
    means each core retains its natural width after resizing, which is then
    used to compute the gap between cores in the stitched image.

    Args:
        crop (Image.Image): The raw cropped core image.
        depth_start (float): Top-of-core depth in metres.
        depth_end (float): Bottom-of-core depth in metres.
        core_strip_height (int): The pixel budget available for a 1 m core.
        max_core_length_m (float): Maximum core length in metres (fills core_strip_height exactly).

    Returns:
        Image.Image: Aspect-ratio-preserved resized core image.
    """
    core_length_m = depth_end - depth_start
    target_height = round((core_length_m / max_core_length_m) * core_strip_height)
    target_height = max(1, target_height)
    aspect = crop.width / crop.height
    target_width = max(1, round(target_height * aspect))
    return crop.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _resize_core_to_width(crop: Image.Image, target_width: int, core_strip_height: int) -> Image.Image:
    """Resize a core crop to a fixed width, preserving aspect ratio.

    Used for outlier cores whose labeled length exceeds max_core_length_m —
    those cores have partial content, so we anchor on width (constant borehole
    diameter) rather than on the unreliable depth label.

    Args:
        crop (Image.Image): The raw cropped core image.
        target_width (int): The desired output width in pixels.
        core_strip_height (int): The pixel budget available for a 1 m core; clamps the output height.

    Returns:
        Image.Image: Aspect-ratio-preserved resized core image.
    """
    aspect = crop.width / crop.height
    target_height = max(1, round(target_width / aspect))

    # clamp max height at core_strip_height to avoid extreme outliers
    if target_height > core_strip_height:
        target_height = core_strip_height
        target_width = max(1, round(target_height * aspect))

    return crop.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _resize_cores(
    cores: list[Image.Image],
    core_height_px: float,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
) -> list[Image.Image]:
    """Resize cores to a common pixel scale, clamping outlier widths to the reference core.

    Args:
        cores (list[Image.Image]): Raw core crops to resize.
        core_height_px (float): Pixel budget for a core spanning core_height_m metres.
        core_height_m (float): Depth extent, in metres, that core_height_px pixels represents.
        core_width_rerror (float): Maximum allowed width ratio vs. the reference core before
            a core is treated as an outlier and rescaled using the reference core's scale factor.

    Returns:
        list[Image.Image]: Cores resized to a consistent pixel scale.
    """
    # To be defined as a reference, the core should be in the same distribution of heights and widths
    widths = np.array([core.width for core in cores])
    heights = np.array([core.height for core in cores])

    error_width = np.abs(widths) / np.median(widths)
    error_height = np.abs(heights) / np.median(heights)
    id_ref = np.argmin(1 - error_width * error_height)

    ref_fx = cores[id_ref].height / (core_height_px * core_height_m)

    cores_resized: list[Image.Image] = []
    for core in cores:
        # Core is within acceptable range
        fx = core.height / (core_height_px * core_height_m)

        if core.width / fx > core_width_rerror * cores[id_ref].width / ref_fx:
            fx = ref_fx

        cores_resized.append(core.resize((round(core.width / fx), round(core.height / fx)), Image.Resampling.LANCZOS))

    # Resize normal cores first so we can derive the target width for outliers
    return cores_resized


def stitch_side_by_side(
    crops: list[Image.Image],
    gap: int,
    padding_vertical: int,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """Place core crops side by side horizontally on a black background.

    Crops are centred horizontally. Each crop is pasted at the top of its
    slot (y = padding_vertical). Cores shorter than CORE_STRIP_HEIGHT leave a
    black gap at the bottom, reflecting partial recovery.

    Args:
        crops (list[Image.Image]): Core strips resized to their natural aspect-ratio widths.
        gap (int): Gap in pixels between adjacent core strips.
        padding_vertical (int): Top and bottom border height in pixels.
        canvas_width (int): The width of the output stitched image.
        canvas_height (int): The height of the output stitched image.

    Returns:
        Image.Image: The stitched image with cores placed side by side.
    """
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    x = _content_x_start(crops, gap, canvas_width)
    for crop in crops:
        canvas.paste(crop, (x, padding_vertical))
        x += crop.width + gap
    return canvas


def stitching_batch(
    cores: list[ImageMetadataProcessed],
    padding_vertical: int = 200,
    padding_horizontal: int = 150,
    ruler_width: int = 300,
    core_height_px: int = 10000,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
    font_size: int = 100,
) -> Image.Image:
    """Stitch core segments together, yielding one output image at a time.

    Each core crop is resized preserving its aspect ratio, scaling its own pixel
    height to match core_height_px * core_height_m (see _resize_cores). Cores whose
    own scale factor would make them disproportionately wide relative to a reference
    core in the same chunk are instead rescaled using the reference core's scale
    factor, keeping pixel density consistent across all cores.

    Note: this scaling is currently based on each crop's raw pixel size, not on its
    depth interval (depth_end - depth_start) — core height is not yet depth-accurate.

    The values are padding_horizontal (A), padding_vertical (B),
    core_height_px (C), and num_cores_per_image (N).

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
    # Laod all cores crops up front so we can identify outliers before resizing
    cores_img = [core.as_image() for core in cores]

    # resize all crops to preserve aspect ratio
    cores_img = _resize_cores(
        cores=cores_img,
        core_height_px=core_height_px,
        core_height_m=core_height_m,
        core_width_rerror=core_width_rerror,
    )

    # Derive gap from remaining horizontal space after placing all cores
    cores_width = sum([core_img.width for core_img in cores_img])
    canvas_width = (
        (3 + len(cores_img)) * padding_horizontal  # Paddings
        + 2 * ruler_width  # Ruler
        + cores_width  # Cores
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
        loc=((2 + len(cores)) * padding_horizontal + ruler_width + cores_width, 3 * padding_vertical),
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
            padding_vertical=padding_vertical,
            padding_horizontal=padding_horizontal,
            ruler_width=ruler_width,
            core_height_px=core_height_px,
            core_height_m=core_height_m,
            core_width_rerror=core_width_rerror,
            font_size=font_size,
        )
