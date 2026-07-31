"""Module for stitching cuttings images together."""

from collections.abc import Generator

from PIL import Image

from src.models import ImageMetadataProcessed
from src.stitching.config import StitchingConfig


def _stitch_cuttings_page(
    cuttings: list[ImageMetadataProcessed],
    config: StitchingConfig,
) -> Image.Image:
    """Arrange one page of cuttings images into a fixed grid, filled column by column.

    Portrait images are rotated 90 degrees to landscape, then scaled down (never up) to fit
    within the grid cell while preserving aspect ratio, and centered in the cell.

    Args:
        cuttings (list[ImageMetadataProcessed]): Cuttings images for this page, in the order
            they should appear (top to bottom within a column, then the next column).
        config (StitchingConfig): Configuration for stitching.

    Returns:
        Image.Image: The stitched page.
    """
    cuttings_config = config.cuttings
    spacing = cuttings_config.spacing
    columns = cuttings_config.num_cuttings_columns
    rows = cuttings_config.num_cuttings_rows
    cell_width = (cuttings_config.output_width - spacing * (columns - 1)) // columns
    cell_height = (cuttings_config.output_height - spacing * (rows - 1)) // rows

    canvas = Image.new("RGB", (cuttings_config.output_width, cuttings_config.output_height))

    for i, cutting in enumerate(cuttings):
        with Image.open(cutting.image_path) as src:
            if src.height > src.width:
                src = src.transpose(Image.Transpose.ROTATE_90)
            scale = min(cell_width / src.width, cell_height / src.height, 1.0)
            img = src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS)

        column, row = divmod(i, rows)
        x = column * (cell_width + spacing) + (cell_width - img.width) // 2
        y = row * (cell_height + spacing) + (cell_height - img.height) // 2
        canvas.paste(img, (x, y))

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
