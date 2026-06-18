"""Module for stitching core segments together."""

from PIL import Image

from src.models import CoreSegmentResult, ImageMetadataProcessed

GAP = 25
PADDING = 80
OUTPUT_WIDTH: int | None = None  # set to resize the final image; None keeps the natural size
OUTPUT_HEIGHT: int | None = None


def cut_core(source: Image.Image, result: CoreSegmentResult) -> Image.Image:
    """Cut a core segment from the source image using the bounding box from the result."""
    left, upper, right, lower = result.bounding_box
    return source.crop((left, upper, right, lower))


def stitch_side_by_side(
    crops: list[Image.Image],
    gap: int,
    padding: int,
    canvas_width: int | None = None,
) -> Image.Image:
    """Place core crops side by side horizontally on a black background.

    gap:          space between adjacent cores.
    padding:      uniform border around the entire image (outside the cores).
    canvas_width: fixes the total output width so all images in a run match.
                  Unoccupied space on the right (before the right padding) is black.
    """
    natural_width = padding + sum(img.width for img in crops) + gap * (len(crops) - 1) + padding
    total_width = canvas_width if canvas_width is not None else natural_width
    total_height = padding + max(img.height for img in crops) + padding
    canvas = Image.new("RGB", (total_width, total_height), color=(0, 0, 0))
    x = padding
    for img in crops:
        canvas.paste(img, (x, padding))
        x += img.width + gap
    return canvas


def stitching(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    gap: int = GAP,
    padding: int = PADDING,
    output_width: int | None = OUTPUT_WIDTH,
    output_height: int | None = OUTPUT_HEIGHT,
    with_mlflow: bool = False,
) -> list[Image.Image]:
    """Stitch core segments together."""
    stitched_images: list[Image.Image] = []
    canvas_width: int | None = None

    for i in range(0, len(imgs), num_cores_per_image):
        chunk = imgs[i : i + num_cores_per_image]
        crops = []
        for meta in chunk:
            src = Image.open(meta.image_path)
            crop = cut_core(src, meta.result)
            crops.append(crop)

        if canvas_width is None:
            canvas_width = padding + crops[0].width * num_cores_per_image + gap * (num_cores_per_image - 1) + padding

        img = stitch_side_by_side(crops, gap=gap, padding=padding, canvas_width=canvas_width)

        if output_width is not None or output_height is not None:
            target_w = output_width or img.width
            target_h = output_height or img.height
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        stitched_images.append(img)

    return stitched_images


# TODO: add depth ruler along the side
# TODO: add start and end depth labels for each core segment
