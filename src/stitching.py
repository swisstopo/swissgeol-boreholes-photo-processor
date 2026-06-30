"""Module for stitching core segments together."""

from collections.abc import Generator

from PIL import Image, ImageDraw, ImageFont

from src.models import CoreSegmentResult, ImageMetadataProcessed


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


def _resize_core_to_width(crop: Image.Image, target_width: int) -> Image.Image:
    """Resize a core crop to a fixed width, preserving aspect ratio.

    Used for outlier cores whose labeled length exceeds max_core_length_m —
    those cores have partial content, so we anchor on width (constant borehole
    diameter) rather than on the unreliable depth label.

    Args:
        crop (Image.Image): The raw cropped core image.
        target_width (int): The desired output width in pixels.

    Returns:
        Image.Image: Aspect-ratio-preserved resized core image.
    """
    aspect = crop.width / crop.height
    target_height = max(1, round(target_width / aspect))
    return crop.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _content_x_start(crops: list[Image.Image], gap: int, canvas_width: int) -> int:
    """Return the x-coordinate where the centred group of crops begins on the canvas.

    Args:
        crops (list[Image.Image]): The list of core crops to be stitched together.
        gap (int): The gap in pixels between adjacent core crops.
        canvas_width (int): The width of the output stitched image.

    Returns:
        int: The x-coordinate where the first core crop should be placed to centre the group.
    """
    total = sum(c.width for c in crops) + gap * max(0, len(crops) - 1)
    return (canvas_width - total) // 2


def _draw_depth_labels(
    img: Image.Image,
    chunk: list[ImageMetadataProcessed],
    crops: list[Image.Image],
    padding_vertical: int,
    gap: int,
) -> Image.Image:
    """Draw depth_start above and depth_end below each individual core strip.

    Args:
        img (Image.Image): The stitched image on which to draw the labels.
        chunk (list[ImageMetadataProcessed]): The list of processed image metadata objects for the cores.
        crops (list[Image.Image]): The list of cropped core images. Includes placeholders for missing cores, but zip
        stops after the real cores are exhausted.
        padding_vertical (int): The vertical padding around the entire image (outside the cores).
        gap (int): The gap between cores in the stitched image.

    Returns:
        Image.Image: The image with depth labels drawn on it.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, padding_vertical // 3))

    x = _content_x_start(crops, gap, img.width)

    for meta, crop in zip(chunk, crops, strict=False):
        cx = x + crop.width // 2
        draw.text(
            (cx, padding_vertical * 3 // 4),
            f"{meta.depth_start:.2f} m",
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        draw.text(
            (cx, img.height - padding_vertical // 2),
            f"{meta.depth_end:.2f} m",
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        x += crop.width + gap
    return img


def _draw_borehole_label(
    img: Image.Image,
    borehole_id: str,
    padding_vertical: int,
    padding_horizontal: int,
) -> Image.Image:
    """Draw the borehole ID in the top-left corner of the image.

    Args:
        img (Image.Image): The stitched image on which to draw the label.
        borehole_id (str): The borehole identifier to display.
        padding_vertical (int): Top and bottom border height in pixels.
        padding_horizontal (int): Left and right border width in pixels.

    Returns:
        Image.Image: The image with the borehole label drawn on it.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, padding_vertical // 3))
    draw.text(
        (padding_horizontal // 4, padding_vertical // 4),
        borehole_id,
        fill=(255, 255, 255),
        font=font,
        anchor="lm",
    )
    return img


def _draw_ruler_label(
    img: Image.Image,
    padding_vertical: int,
    padding_horizontal: int,
) -> Image.Image:
    """Draw a ruler label at the specified position.

    Args:
        img (Image.Image): The image on which to draw the label.
        padding_vertical (int): Top and bottom border height in pixels.
        padding_horizontal (int): Left and right border width in pixels.

    Returns:
        Image.Image: The image with the ruler label drawn on it.
    """
    return img  # placeholder


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


def stitching(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    padding_vertical: int = 95,
    padding_horizontal: int = 110,
    output_width: int = 1144,
    output_height: int = 1260,
    max_core_length_m: float = 1.0,
) -> Generator[Image.Image, None, None]:
    """Stitch core segments together, yielding one output image at a time.

    Each core is resized preserving its aspect ratio, with height proportional
    to its depth extent (depth_end - depth_start) relative to max_core_length_m.
    Cores whose depth extent exceeds max_core_length_m are treated as outliers
    (partial recovery in an oversized box) and are instead width-matched to the
    average width of the normal cores in the same chunk, keeping pixel density
    consistent across all cores.

    The gap between cores is derived from the remaining horizontal space after
    placing all cores and side padding:

        gap = (output_width - 2 * padding_horizontal - sum(core widths)) / (n - 1)

    This is a generator: it yields each stitched image as soon as it is ready
    instead of building a list of all results in memory.

    Typical usage::
        for img in stitching(cores):
            img.save("output.png")

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): The number of cores to place side by side in each stitched image.
        padding_vertical (int): Top and bottom border height in pixels.
        padding_horizontal (int): Left and right border width in pixels.
        output_width (int): The canvas width. The gap between cores is derived from the remaining space.
        output_height (int): The canvas height.
        max_core_length_m (float): Maximum core length in metres (fills the strip height exactly).
            Cores exceeding this are width-matched to normal cores in the same chunk.

    Yields:
        Image.Image: One stitched image per chunk of up to num_cores_per_image cores.
    """
    for i in range(0, len(imgs), num_cores_per_image):
        # core_strip_height is the pixel budget available for a 1 m core.
        # Derived from the fixed canvas height minus top and bottom padding.
        core_strip_height = output_height - 2 * padding_vertical

        # Process a chunk of up to num_cores_per_image cores
        chunk = imgs[i : i + num_cores_per_image]

        # Cut all raw crops up front so we can identify outliers before resizing
        raw: list[tuple[ImageMetadataProcessed, Image.Image]] = []
        for meta in chunk:
            with Image.open(meta.image_path) as src:
                raw.append((meta, _cut_core(src, meta.result)))

        is_outlier = [(meta.depth_end - meta.depth_start) > max_core_length_m for meta, _ in raw]

        # Resize normal cores first so we can derive the target width for outliers
        normal_crops = [
            _resize_core(crop, meta.depth_start, meta.depth_end, core_strip_height, max_core_length_m)
            for (meta, crop), outlier in zip(raw, is_outlier, strict=True)
            if not outlier
        ]
        avg_normal_width = (
            round(sum(c.width for c in normal_crops) / len(normal_crops)) if normal_crops else core_strip_height // 8
        )

        # Build the final crops list in original order; outliers are width-matched
        crops: list[Image.Image] = []
        normal_iter = iter(normal_crops)
        for (_, crop), outlier in zip(raw, is_outlier, strict=True):
            if outlier:
                crops.append(_resize_core_to_width(crop, avg_normal_width))
            else:
                crops.append(next(normal_iter))

        # Pad with black placeholders so every output image has the same layout.
        # Width is the average of the real crops so the layout stays consistent.
        if len(crops) < num_cores_per_image:
            avg_core_width = round(sum(c.width for c in crops) / len(crops))
            placeholder = Image.new("RGB", (avg_core_width, core_strip_height), color=(0, 0, 0))
            crops += [placeholder] * (num_cores_per_image - len(crops))

        # Derive gap from remaining horizontal space after placing all cores
        total_cores_width = sum(c.width for c in crops)
        gap = (
            max(0, (output_width - 2 * padding_horizontal - total_cores_width) // (num_cores_per_image - 1))
            if num_cores_per_image > 1
            else 0
        )

        # place cores side by side on a black canvas and draw depth labels
        img = stitch_side_by_side(
            crops,
            gap=gap,
            padding_vertical=padding_vertical,
            canvas_width=output_width,
            canvas_height=output_height,
        )
        # Pass all crops (real + placeholders) so x-start matches stitch_side_by_side;
        # strict=False stops the zip after the real cores are exhausted.
        img = _draw_depth_labels(
            img,
            chunk=chunk,
            crops=crops,
            padding_vertical=padding_vertical,
            gap=gap,
        )
        img = _draw_borehole_label(
            img,
            borehole_id=chunk[0].borehole_id,
            padding_vertical=padding_vertical,
            padding_horizontal=padding_horizontal,
        )
        img = _draw_ruler_label(img, padding_vertical=padding_vertical, padding_horizontal=padding_horizontal)

        yield img
