"""Module for stitching cuttings images together."""

import logging
from dataclasses import dataclass

from PIL import Image

from src.models import ImageMetadataProcessedCuttings
from src.stitching.config import StitchingConfig
from src.stitching.draw import _draw_borehole_label, _draw_cuttings_annotation, _draw_cuttings_border_label

logger = logging.getLogger(__name__)


@dataclass
class StitchingBatchCuttings:
    """One page of cuttings plus the values shared across all pages, for parallel dispatch."""

    cuttings: list[ImageMetadataProcessedCuttings]  # cuttings assigned to this page
    shared_borehole_id: str  # borehole ID drawn in the label, shared across all pages


def stitching_batch_cuttings(
    cuttings: list[ImageMetadataProcessedCuttings],
    shared_borehole_id: str,
    config: StitchingConfig,
) -> Image.Image:
    """Stitch one batch (page) of cuttings images into a single output image.

    Cuttings are arranged into a fixed grid, filled column by column. Portrait images are
    rotated 90 degrees to landscape, then scaled down (never up) to fit within the grid cell
    while preserving aspect ratio, and right-aligned so the gap to the depth annotation stays
    constant regardless of the scaled-down image's width.

    The values are padding_horizontal (PH), padding_vertical (PV),
    and padding cuttings (PC). FROM/TO show the depth_start of the topmost/bottommost
    cutting in each column, centered in the top/bottom PV border.

    <-----------------------------------------WIDTH---------------------------------------->
    ----------------------------------------------------------------------------------------    ʌ
    | ID   ʌ                                                                               |    |
    |     PV  FROM                     FROM                     FROM                       |    |
    |      v                                                                               |    |
    | <PH> |---------| <annotation> |---------| <annotation> |---------| <annotation> <PH> |    |
    |      | CUT 1.1 |              | CUT 2.1 |              | CUT 3.1 |                   |    |
    |      |---------|              |---------|              |---------|                   |    |
    |           ʌ                        ʌ                        ʌ                        |    |
    |           PC                       PC                       PC                       |    |
    |           v                        v                        v                        |   HEIGHT
    |      |---------|              |---------|              |---------|                   |    |
    |      | CUT 1.2 |              | CUT 2.2 |              | CUT 3.2 |                   |    |
    |      |---------|              |---------|              |---------|                   |    |
    |           .                        .                        .                        |    |
    |           .                        .                        .                        |    |
    |           .                        .                        .                        |    |
    |      ʌ                                                                               |    |
    |     PV   TO                       TO                       TO                        |    |
    |      v                                                                               |    |
    ---------------------------------------------------------------------------------------     v

    Args:
        cuttings (list[ImageMetadataProcessedCuttings]): Cuttings images for this batch, in the order
            they should appear (top to bottom within a column, then the next column).
        shared_borehole_id (str): Borehole ID drawn in the top-left label, shared across all batches.
        config (StitchingConfig): Configuration for stitching.

    Returns:
        Image.Image: The stitched image for this batch of cuttings.
    """
    cuttings_config = config.cuttings
    columns = cuttings_config.num_cuttings_columns
    rows = cuttings_config.num_cuttings_rows
    # reserve top (ID+FROM band) and bottom (TO band) padding, plus gaps between rows
    reserved_height = 2 * cuttings_config.padding_vertical + cuttings_config.padding_cuttings * (rows - 1)
    cell_height = (cuttings_config.output_height - reserved_height) // rows
    # each column reserves PH before the image, annotation_gap between the image and its
    # annotation, and (shared with the next column) PH after the annotation
    reserved_width = cuttings_config.padding_horizontal * (columns + 1) + cuttings_config.annotation_gap * columns
    content_width = (cuttings_config.output_width - reserved_width) // columns
    image_width = content_width - cuttings_config.annotation_width
    column_step = (
        image_width
        + cuttings_config.annotation_gap
        + cuttings_config.annotation_width
        + cuttings_config.padding_horizontal
    )

    cutting_imgs = []
    for cutting in cuttings:
        src = cutting.load_cuttings()
        # scale down (never up) to fit the cell while preserving aspect ratio
        scale = min(image_width / src.width, cell_height / src.height, 1.0)
        img = src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS)
        cutting_imgs.append(img)
        # each cutting belongs to exactly one page, so its full-resolution cache can be freed once
        # resized here -- otherwise every cutting's crop stays cached in memory for the whole run
        cutting.release_cuttings_cache()

    canvas = Image.new("RGB", (cuttings_config.output_width, cuttings_config.output_height), color=(0, 0, 0))

    # ID and FROM share the single padding_vertical band, stacked close together (top/bottom quarters)
    canvas = _draw_borehole_label(
        canvas,
        borehole_id=shared_borehole_id,
        loc=(100, round(cuttings_config.padding_vertical / 4)),
        font_size=cuttings_config.font_size,
    )

    for i, cutting_img in enumerate(cutting_imgs):
        # column-major fill: cuttings 0..rows-1 go in column 0, rows..2*rows-1 in column 1, etc.
        column, row = divmod(i, rows)
        image_x = cuttings_config.padding_horizontal + column * column_step
        cell_y = cuttings_config.padding_vertical + row * (cell_height + cuttings_config.padding_cuttings)
        y = cell_y + (cell_height - cutting_img.height) // 2
        # right-aligned (not centered/left-aligned) so the gap to the annotation is always exactly
        # annotation_gap, regardless of how narrow the aspect-scaled image ends up
        canvas.paste(cutting_img, (image_x + image_width - cutting_img.width, y))

        annotation_x = image_x + image_width + cuttings_config.annotation_gap
        canvas = _draw_cuttings_annotation(
            canvas,
            depth=cuttings[i].depth,
            loc=(annotation_x, cell_y),
            size=(cuttings_config.annotation_width, cell_height),
            font_size=cuttings_config.font_size,
        )

    # top/bottom border labels: depth_start of the first and last cutting in each column
    for column in range(columns):
        start_idx = column * rows
        # stop once a column has no cuttings (the last page may not fill every column)
        if start_idx >= len(cuttings):
            break
        end_idx = min(start_idx + rows, len(cuttings)) - 1
        image_right = cuttings_config.padding_horizontal + column * column_step + image_width

        # right-anchored at image_right (not centered on image_width) since cuttings are
        # right-aligned in their column and vary in width, so the right edge is the only
        # x position shared by every cutting in the column
        canvas = _draw_cuttings_border_label(
            canvas,
            depth=cuttings[start_idx].depth,
            loc=(image_right, round(cuttings_config.padding_vertical * 3 / 4)),
            font_size=cuttings_config.font_size,
            anchor="rm",
        )
        canvas = _draw_cuttings_border_label(
            canvas,
            depth=cuttings[end_idx].depth,
            loc=(image_right, round(cuttings_config.output_height - cuttings_config.padding_vertical / 2)),
            font_size=cuttings_config.font_size,
            anchor="rm",
        )

    return canvas


def stitching_cuttings(
    imgs: list[ImageMetadataProcessedCuttings],
    config: StitchingConfig,
) -> list[StitchingBatchCuttings]:
    """Split cuttings images into pages of a fixed grid, ready to be stitched.

    Args:
        imgs (list[ImageMetadataProcessedCuttings]): Cuttings images to stitch, in the order they should
            appear on the page (top to bottom within a column, then the next column).
        config (StitchingConfig): Configuration for stitching.

    Returns:
        list[StitchingBatchCuttings]: One batch per page of up to
            (num_cuttings_columns * num_cuttings_rows) cuttings.
    """
    if not imgs:
        logger.warning("No cuttings images to stitch")
        return []

    num_cuttings_per_page = config.cuttings.num_cuttings_columns * config.cuttings.num_cuttings_rows

    return [
        StitchingBatchCuttings(
            cuttings=imgs[i : i + num_cuttings_per_page],
            shared_borehole_id=imgs[0].borehole_id,
        )
        for i in range(0, len(imgs), num_cuttings_per_page)
    ]
