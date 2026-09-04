"""Drawing helpers for composing stitched core and cuttings images: placement, labels, and rulers."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _format_depth(value: float) -> str:
    """Format a depth with a trailing "m", an apostrophe thousands separator, and no uninformative ".00"."""
    rounded = round(value, 2)
    text = f"{int(rounded):,}" if rounded == int(rounded) else f"{rounded:,.2f}"
    return text.replace(",", "'") + "m"


def _draw_cores(
    canvas: Image.Image,
    cores: list[Image.Image],
    labels_range: list[tuple[float, float]],
    is_scale_correct: list[bool],
    loc: tuple[int, int],
    padding_horizontal: int,
    padding_vertical: int,
    max_core_height: int,
    font_size: int,
) -> Image.Image:
    """Paste core crops onto the canvas and draw their depth labels above/below.

    Args:
        canvas (Image.Image): The canvas to draw onto.
        cores (list[Image.Image]): Resized core crops to place, left to right.
        labels_range (list[tuple[float, float]]): (depth_start, depth_end) per core, same order as cores.
        is_scale_correct (list[bool]): Indicate if the scale of the displayed core is to be trusted.
        loc (tuple[int, int]): Top-left corner of the first core.
        padding_horizontal (int): Horizontal gap between cores in pixels.
        padding_vertical (int): Vertical space reserved for labels above/below each core.
        max_core_height (int): Row height in pixels reserved for a core; labels are drawn
            padding_vertical pixels above/below this band.
        font_size (int): Font size used for the depth labels.

    Returns:
        Image.Image: The canvas with cores and labels drawn on it.
    """
    x_min, y_min = loc
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)

    for i, (core, label_range, correct) in enumerate(zip(cores, labels_range, is_scale_correct, strict=True)):
        x_offset = x_min + i * padding_horizontal + sum(c.width for c in cores[:i])
        start_label, end_label = label_range
        canvas.paste(core, (x_offset, y_min))

        draw.text(
            (x_offset + core.width / 2, y_min - padding_vertical),
            _format_depth(start_label) + ("" if correct else "*"),
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        draw.text(
            (x_offset + core.width / 2, y_min + max_core_height + padding_vertical),
            _format_depth(end_label) + ("" if correct else "*"),
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )

    return canvas


def _draw_borehole_label(
    canvas: Image.Image,
    borehole_id: str,
    loc: tuple[int, int],
    font_size: int,
) -> Image.Image:
    """Draw the borehole ID in the top-left corner of the image.

    Args:
        canvas (Image.Image): The stitched image on which to draw the label.
        borehole_id (str): The borehole identifier to display.
        loc (tuple[int, int]): Position of the label (left-middle anchor).
        font_size (int): Font size used for the label.

    Returns:
        Image.Image: The image with the borehole label drawn on it.
    """
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    draw.text(
        (loc[0], loc[1]),
        borehole_id,
        fill=(255, 255, 255),
        font=font,
        anchor="lm",
    )
    return canvas


def _draw_cuttings_annotation(
    canvas: Image.Image,
    depth: float,
    loc: tuple[int, int],
    size: tuple[int, int],
    font_size: int,
) -> Image.Image:
    """Draw a cutting's depth label onto its own strip and paste it onto the canvas.

    Drawn onto its own strip first (rather than directly on the canvas) so a label wider than
    the strip is clipped instead of bleeding into the padding to its right.

    Args:
        canvas (Image.Image): The canvas to paste the label onto.
        depth (float): Depth value to display.
        loc (tuple[int, int]): Top-left corner of the label strip.
        size (tuple[int, int]): (width, height) of the label strip in pixels.
        font_size (int): Font size used for the label.

    Returns:
        Image.Image: The canvas with the label pasted onto it.
    """
    annotation_img = Image.new("RGB", size, color=(0, 0, 0))
    font = ImageFont.load_default(size=font_size)
    ImageDraw.Draw(annotation_img).text(
        (size[0] / 2, size[1] / 2),
        _format_depth(depth),
        fill=(255, 255, 255),
        font=font,
        anchor="mm",
    )
    canvas.paste(annotation_img, loc)
    return canvas


def _draw_cuttings_border_label(
    canvas: Image.Image,
    depth: float,
    loc: tuple[int, int],
    font_size: int,
    anchor: str = "lm",
) -> Image.Image:
    """Draw a column's top/bottom depth label (FROM/TO) at loc.

    Args:
        canvas (Image.Image): The canvas to draw onto.
        depth (float): Depth value to display.
        loc (tuple[int, int]): Position of the label.
        font_size (int): Font size used for the label.
        anchor (str): PIL text anchor, e.g. "lm" (left-middle) or "rm" (right-middle).

    Returns:
        Image.Image: The canvas with the label drawn on it.
    """
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    draw.text(
        loc,
        _format_depth(depth),
        fill=(255, 255, 255),
        font=font,
        anchor=anchor,
    )
    return canvas


def _draw_ruler(
    img: Image.Image,
    loc: tuple[int, int],
    size: tuple[int, int],
    n_major: int = 100,
    n_intermediate: int = 2,
    n_minor: int = 10,
    font_size: int = 100,
    horizontal_flip: bool = False,
) -> Image.Image:
    """Draw a vertical depth ruler with major/intermediate/minor ticks.

    Args:
        img (Image.Image): The stitched image on which to draw the ruler.
        loc (tuple[int, int]): Top-left corner of the ruler.
        size (tuple[int, int]): (width, height) of the ruler area in pixels.
        n_major (int): Number of major ticks (labeled).
        n_intermediate (int): Number of intermediate ticks per major-tick interval.
        n_minor (int): Number of minor ticks per major-tick interval.
        font_size (int): Font size for major tick labels.
        horizontal_flip (bool): If True, mirror tick lengths and label anchor for a ruler on the right edge.

    Returns:
        Image.Image: The image with the ruler drawn on it.
    """
    x_start, y_start = loc
    x_end, y_end = x_start + size[0], y_start + size[1]

    if horizontal_flip:
        x_start, x_end = (x_end, x_start)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size)

    w_span = x_end - x_start
    major_ticks = np.linspace(start=y_start, stop=y_end, num=n_major + 1, endpoint=True)
    intermediate_ticks = np.linspace(start=y_start, stop=y_end, num=n_major * n_intermediate + 1, endpoint=True)
    minor_ticks = np.linspace(start=y_start, stop=y_end, num=n_major * n_minor + 1, endpoint=True)

    for minor_tick in minor_ticks:
        draw.line((x_start, minor_tick) + (x_start + 0.4 * w_span, minor_tick), width=1, fill=(255, 255, 255))

    for inter_ticks in intermediate_ticks:
        draw.line((x_start, inter_ticks) + (x_start + 0.6 * w_span, inter_ticks), width=1, fill=(255, 255, 255))

    for i, major_tick in enumerate(major_ticks):
        draw.line((x_start, major_tick) + (x_start + 0.8 * w_span, major_tick), width=1, fill=(255, 255, 255))
        draw.text(
            (x_start + 0.9 * w_span, major_tick),
            f"{i}",
            fill=(255, 255, 255),
            font=font,
            anchor="lm" if not horizontal_flip else "rm",
        )
    return img
