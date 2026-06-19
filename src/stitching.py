"""Module for stitching core segments together."""

from collections.abc import Generator

from PIL import Image, ImageDraw, ImageFont

from src.models import CoreSegmentResult, ImageMetadataProcessed

NUM_CORES_PER_IMAGE = 6
PADDING = 80
CORE_WIDTH = 140  # assumed width of a segmented borehole core in pixels
OUTPUT_WIDTH = 1144
OUTPUT_HEIGHT = 1260


def cut_core(source: Image.Image, result: CoreSegmentResult) -> Image.Image:
    """Cut a core segment from the source image using the bounding box from the result.

    Args:
        source (Image.Image): The source image from which to cut the core segment.
        result (CoreSegmentResult): The result containing the bounding box for the core segment.

    Returns:
        Image.Image: The cropped core segment image.
    """
    left, upper, right, lower = result.bounding_box
    return source.crop((left, upper, right, lower))


def _draw_depth_labels(
    img: Image.Image,
    chunk: list[ImageMetadataProcessed],
    crop_widths: list[int],
    padding: int,
    gap: int,
) -> None:
    """Draw depth_start above and depth_end below each individual core strip.

    Args:
        img (Image.Image): The stitched image on which to draw the labels.
        chunk (list[ImageMetadataProcessed]): The list of processed image metadata objects for the cores.
        crop_widths (list[int]): The widths of the cropped core images.
        padding (int): The uniform border around the entire image (outside the cores).
        gap (int): The gap between cores in the stitched image.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, padding // 3))
    total_content_width = CORE_WIDTH * len(crop_widths) + gap * max(0, len(crop_widths) - 1)
    x = (img.width - total_content_width) // 2
    for meta, width in zip(chunk, crop_widths, strict=False):
        cx = x + width // 2
        draw.text((cx, padding * 3 // 4), f"{meta.depth_start:.2f} m", fill=(255, 255, 255), font=font, anchor="mm")
        draw.text(
            (cx, img.height - padding // 2), f"{meta.depth_end:.2f} m", fill=(255, 255, 255), font=font, anchor="mm"
        )
        x += CORE_WIDTH + gap


def _draw_borehole_label(img: Image.Image, borehole_id: str, padding: int) -> None:
    """Draw the borehole ID in the top-left corner of the image.

    Args:
        img (Image.Image): The stitched image on which to draw the label.
        borehole_id (str): The borehole identifier to display.
        padding (int): The uniform border around the entire image (outside the cores).
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, padding // 3))
    draw.text((padding // 4, padding // 4), borehole_id, fill=(255, 255, 255), font=font, anchor="lm")


def _draw_rulerlabel(
    img: Image.Image,
) -> None:
    """Draw a ruler label at the specified position.

    Args:
        img (Image.Image): The image on which to draw the label.
    """
    pass  # placeholder for drawing a ruler label


def stitch_side_by_side(
    crops: list[Image.Image],
    gap: int,
    padding: int,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """Place core crops side by side horizontally on a black background.

    Args:
        crops (list[Image.Image]): The list of cropped core images to stitch together.
        gap (int): The gap between cores in the stitched image.
        padding (int): The uniform border around the entire image (outside the cores).
        canvas_width (int): The width of the output stitched image.
        canvas_height (int): The height of the output stitched image.

    Returns:
        Image.Image: The stitched image with cores placed side by side.
    """
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    total_content_width = CORE_WIDTH * len(crops) + gap * max(0, len(crops) - 1)
    x = (canvas_width - total_content_width) // 2
    for img in crops:
        canvas.paste(img, (x, padding))
        x += CORE_WIDTH + gap
    return canvas


def stitching(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = NUM_CORES_PER_IMAGE,
    padding: int = PADDING,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
) -> Generator[Image.Image, None, None]:
    """Stitch core segments together, yielding one output image at a time.

    This is a generator: it yields each stitched image as soon as it is ready
    instead of building a list of all results in memory. This keeps memory usage
    constant regardless of how many output images are produced — the caller should
    save or process each image before requesting the next one.

    Typical usage::
        for img in stitch(cores):
            img.save("output.png")

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): The number of cores to place side by side in each stitched image.
        padding (int): The uniform border around the entire image (outside the cores).
        output_width (int): The canvas width. The gap between cores is derived from the remaining space.
        output_height (int): The canvas height.

    Yields:
        Image.Image: One stitched image per chunk of up to num_cores_per_image cores.
    """
    for i in range(0, len(imgs), num_cores_per_image):
        # Process a chunk of up to num_cores_per_image cores
        chunk = imgs[i : i + num_cores_per_image]
        crops = []
        for meta in chunk:
            src = Image.open(meta.image_path)
            crop = cut_core(src, meta.result)
            crops.append(crop)

        # Pad with black placeholders so every output image has the same num_cores_per_image layout
        placeholder_height = output_height - 2 * padding
        for _ in range(num_cores_per_image - len(crops)):
            crops.append(Image.new("RGB", (CORE_WIDTH, placeholder_height), color=(0, 0, 0)))

        # calculate gap based on remaining space after placing cores and padding
        gap = (
            max(0, (output_width - 2 * padding - CORE_WIDTH * num_cores_per_image) // (num_cores_per_image - 1))
            if num_cores_per_image > 1
            else 0
        )

        # place cores side by side on a black canvas and draw depth labels
        img = stitch_side_by_side(
            crops, gap=gap, padding=padding, canvas_width=output_width, canvas_height=output_height
        )
        # Pass all crop widths (including placeholders) so x-start matches stitch_side_by_side;
        # zip stops at len(chunk) so only real cores get labels.
        _draw_depth_labels(img, chunk=chunk, crop_widths=[c.width for c in crops], padding=padding, gap=gap)
        _draw_borehole_label(img, borehole_id=chunk[0].borehole_id, padding=padding)
        _draw_rulerlabel(img)  # placeholder

        yield img
