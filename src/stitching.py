"""Module for stitching core segments together."""

from collections.abc import Generator

from PIL import Image, ImageDraw, ImageFont

from src.models import CoreSegmentResult, ImageMetadataProcessed

PADDING = 90
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
    x = padding
    for meta, width in zip(chunk, crop_widths, strict=True):
        cx = x + width // 2
        draw.text((cx, padding // 2), f"{meta.depth_start:.2f} m", fill=(255, 255, 255), font=font, anchor="mm")
        draw.text(
            (cx, img.height - padding // 2), f"{meta.depth_end:.2f} m", fill=(255, 255, 255), font=font, anchor="mm"
        )
        x += width + gap


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
    x = padding
    for img in crops:
        canvas.paste(img, (x, padding))
        x += img.width + gap
    return canvas


def stitching(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    padding: int = PADDING,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
    with_mlflow: bool = False,
) -> Generator[Image.Image, None, None]:
    """Stitch core segments together, yielding one output image at a time.

    This is a generator: it yields each stitched image as soon as it is ready
    instead of building a list of all results in memory. This keeps memory usage
    constant regardless of how many output images are produced — the caller should
    save or process each image before requesting the next one.

    Typical usage::
        for img in stitching(cores):
            img.save("output.png")

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): The number of cores to place side by side in each stitched image.
        padding (int): The uniform border around the entire image (outside the cores).
        output_width (int): The canvas width. The gap between cores is derived from the remaining space.
        output_height (int): The canvas height.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Yields:
        Image.Image: One stitched image per chunk of up to num_cores_per_image cores.
    """
    for i in range(0, len(imgs), num_cores_per_image):
        chunk = imgs[i : i + num_cores_per_image]
        crops = []
        for meta in chunk:
            src = Image.open(meta.image_path)
            crop = cut_core(src, meta.result)
            crops.append(crop)

        full_batch_gaps = num_cores_per_image - 1
        gap = (
            max(0, (output_width - 2 * padding - crops[0].width * num_cores_per_image) // full_batch_gaps)
            if full_batch_gaps > 0
            else 0
        )

        img = stitch_side_by_side(
            crops, gap=gap, padding=padding, canvas_width=output_width, canvas_height=output_height
        )
        _draw_depth_labels(img, chunk=chunk, crop_widths=[c.width for c in crops], padding=padding, gap=gap)

        yield img


# TODO: add mlflow tracking
