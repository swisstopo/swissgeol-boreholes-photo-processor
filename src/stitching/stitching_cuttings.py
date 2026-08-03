"""Module for stitching cuttings images together."""

from collections.abc import Generator

from PIL import Image, ImageDraw, ImageFont

from src.models import ImageMetadataProcessed
from src.stitching.config import StitchingConfig


def _stitch_cuttings_page(
    cuttings: list[ImageMetadataProcessed],
    config: StitchingConfig,
) -> Image.Image:
    """Arrange one page of cuttings images into a fixed grid, filled column by column.

    Portrait images are rotated 90 degrees to landscape, then scaled down (never up) to fit
    within the grid cell while preserving aspect ratio, and centered in the cell.

    The values are padding_horizontal (PH), padding_vertical (PV),
    and padding cuttings (PC). FROM/TO show the depth_start of the topmost/bottommost
    cutting in each column, centered in the top/bottom PV border.

    <-------------------------------------------2084--------------------------------------->
    ----------------------------------------------------------------------------------------    ʌ
    | ID   ʌ                                                                               |    |
    |     PV  FROM                     FROM                     FROM                       |    |
    |      v                                                                               |    |
    | <PH> |---------| <annotation> |---------| <annotation> |---------| <annotation> <PH> |    |
    |      | CUT 1.1 |              | CUT 2.1 |              | CUT 3.1 |                   |    |
    |      |---------|              |---------|              |---------|                   |    |
    |          ʌ                         ʌ                        ʌ                        |    |
    |          PC                        PC                       PC                       |    |
    |          v                         v                        v                        |   2896
    |      |---------|              |---------|              |---------|                   |    |
    |      | CUT 1.2 |              | CUT 2.2 |              | CUT 3.2 |                   |    |
    |      |---------|              |---------|              |---------|                   |    |
    |           .                        .                        .                        |    |
    |           .                        .                        .                        |    |
    |           .                        .                        .                        |    |
    |      ʌ                                                                               |    |
    |     PV   TO                       TO                             TO                  |    |
    |      v                                                                               |    |
    ---------------------------------------------------------------------------------------     v

    Args:
        cuttings (list[ImageMetadataProcessed]): Cuttings images for this page, in the order
            they should appear (top to bottom within a column, then the next column).
        config (StitchingConfig): Configuration for stitching.

    Returns:
        Image.Image: The stitched page.
    """
    cuttings_config = config.cuttings
    padding_cuttings = cuttings_config.padding_cuttings
    padding_horizontal = cuttings_config.padding_horizontal
    padding_vertical = cuttings_config.padding_vertical
    annotation_width = cuttings_config.annotation_width
    annotation_gap = cuttings_config.annotation_gap
    columns = cuttings_config.num_cuttings_columns
    rows = cuttings_config.num_cuttings_rows
    cell_height = (cuttings_config.output_height - 2 * padding_vertical - padding_cuttings * (rows - 1)) // rows
    # each column reserves PH before the image, annotation_gap between the image and its
    # annotation, and (shared with the next column) PH after the annotation
    reserved_width = padding_horizontal * (columns + 1) + annotation_gap * columns
    content_width = (cuttings_config.output_width - reserved_width) // columns
    image_width = content_width - annotation_width
    column_step = image_width + annotation_gap + annotation_width + padding_horizontal

    # rotate all images to landscape
    cutting_imgs = []
    for _, cutting in enumerate(cuttings):
        with Image.open(cutting.image_path) as src:
            if src.height > src.width:
                src = src.transpose(Image.Transpose.ROTATE_90)
            scale = min(image_width / src.width, cell_height / src.height, 1.0)
            img = src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS)
            cutting_imgs.append(img)

    canvas = Image.new("RGB", (cuttings_config.output_width, cuttings_config.output_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=cuttings_config.font_size)

    for i, cutting_img in enumerate(cutting_imgs):
        column, row = divmod(i, rows)
        image_x = padding_horizontal + column * column_step
        cell_y = padding_vertical + row * (cell_height + padding_cuttings)
        # left-aligned (not centered) so every row's left margin is exactly PH, matching the
        # fixed PH gap after the last column's annotation on the right
        y = cell_y + (cell_height - cutting_img.height) // 2
        canvas.paste(cutting_img, (image_x, y))

        cutting = cuttings[i]
        # drawn onto its own strip and pasted, so text wider than annotation_width is clipped
        # rather than bleeding into the PH gap to its right
        annotation_img = Image.new("RGB", (annotation_width, cell_height), color=(0, 0, 0))
        ImageDraw.Draw(annotation_img).text(
            (annotation_width / 2, cell_height / 2),
            f"{cutting.depth_start:.2f}",
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
            stroke_width=cuttings_config.font_stroke_width,
        )
        annotation_x = image_x + image_width + annotation_gap
        canvas.paste(annotation_img, (annotation_x, cell_y))

    # top/bottom border labels: depth_start of the first and last cutting in each column
    for column in range(columns):
        start_idx = column * rows
        if start_idx >= len(cuttings):
            break
        end_idx = min(start_idx + rows, len(cuttings)) - 1
        image_x = padding_horizontal + column * column_step

        # left-anchored at image_x (not centered on image_width) since cuttings are
        # left-aligned in their column and vary in width, so the left edge is the only
        # x position shared by every cutting in the column
        draw.text(
            (image_x, padding_vertical / 2),
            f"{cuttings[start_idx].depth_start:.2f}",
            fill=(255, 255, 255),
            font=font,
            anchor="lm",
            stroke_width=cuttings_config.font_stroke_width,
        )
        draw.text(
            (image_x, cuttings_config.output_height - padding_vertical / 2),
            f"{cuttings[end_idx].depth_start:.2f}",
            fill=(255, 255, 255),
            font=font,
            anchor="lm",
            stroke_width=cuttings_config.font_stroke_width,
        )

    return canvas


def stitching_cuttings(
    imgs: list[ImageMetadataProcessed],
    config: StitchingConfig,
) -> Generator[Image.Image, None, None]:
    """Stitch cuttings images into pages of a fixed grid, one output image per page.

    Args:
        imgs (list[ImageMetadataProcessed]): Cuttings images to stitch, in the order they should
            appear on the page (top to bottom within a column, then the next column).
        config (StitchingConfig): Configuration for stitching.

    Yields:
        Image.Image: One stitched page per chunk of (num_cuttings_columns * num_cuttings_rows) images.
    """
    # As a start we will just go through the folder in order
    # in the future this needs to be sorted by depth and then stitched together
    num_cuttings_per_page = config.cuttings.num_cuttings_columns * config.cuttings.num_cuttings_rows

    for i in range(0, len(imgs), num_cuttings_per_page):
        yield _stitch_cuttings_page(
            cuttings=imgs[i : i + num_cuttings_per_page],
            config=config,
        )
