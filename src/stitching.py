"""Module for stitching core segments together."""

from PIL import Image, ImageDraw, ImageFont

from src.models import CoreSegmentResult, ImageMetadataProcessed

PADDING = 85
OUTPUT_WIDTH = 1144
OUTPUT_HEIGHT = 1260


def cut_core(source: Image.Image, result: CoreSegmentResult) -> Image.Image:
    """Cut a core segment from the source image using the bounding box from the result."""
    left, upper, right, lower = result.bounding_box
    return source.crop((left, upper, right, lower))


def _draw_depth_labels(
    img: Image.Image,
    chunk: list[ImageMetadataProcessed],
    crop_widths: list[int],
    padding: int,
    gap: int,
) -> None:
    """Draw depth_start above and depth_end below each individual core strip."""
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
    """Place core crops side by side horizontally on a black background."""
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
) -> list[Image.Image]:
    """Stitch core segments together.

    Args:
        imgs (list[ImageMetadataProcessed]): The list of processed image metadata objects to stitch together.
        num_cores_per_image (int): The number of cores to place side by side in each stitched image.
        padding (int): The uniform border around the entire image (outside the cores).
        output_width (int): The canvas width. The gap between cores is derived from the remaining space.
        output_height (int): The canvas height.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[Image.Image]: A list of stitched images, each containing up to num_cores_per_image cores.
    """
    stitched_images: list[Image.Image] = []

    for i in range(0, len(imgs), num_cores_per_image):
        chunk = imgs[i : i + num_cores_per_image]
        crops = []
        for meta in chunk:
            src = Image.open(meta.image_path)
            crop = cut_core(src, meta.result)
            crops.append(crop)

        num_gaps = len(crops) - 1
        total_crop_width = sum(c.width for c in crops)
        gap = max(0, (output_width - 2 * padding - total_crop_width) // num_gaps) if num_gaps > 0 else 0

        img = stitch_side_by_side(
            crops, gap=gap, padding=padding, canvas_width=output_width, canvas_height=output_height
        )
        _draw_depth_labels(img, chunk=chunk, crop_widths=[c.width for c in crops], padding=padding, gap=gap)

        stitched_images.append(img)

    return stitched_images


# TODO: add mlflow tracking
