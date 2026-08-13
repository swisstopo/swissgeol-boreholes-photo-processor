"""Tests for the stitching_cuttings module."""

import numpy as np
import pytest
from PIL import Image

from src.models import CuttingsSegmentResult, ImageMetadataCuttings, ImageMetadataProcessedCuttings
from src.stitching.config import CuttingsStitchingConfig, StitchingConfig
from src.stitching.stitching_cuttings import stitching_cuttings

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


@pytest.fixture
def make_cutting(tmp_path):
    def _factory(
        depth: float,
        size: tuple[int, int] = (80, 60),
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> ImageMetadataProcessedCuttings:
        """Creates a simple ImageMetadataProcessedCuttings for a cutting with a solid-color image."""
        image_path = tmp_path / f"cutting_{depth:07.2f}.png"
        Image.new("RGB", size, color=color).save(image_path)
        metadata = ImageMetadataCuttings(
            borehole_id="GES-F-1",
            depth=depth,
            image_path=image_path,
        )
        cuttings = CuttingsSegmentResult(bbox=(0.0, 0.0, float(size[0]), float(size[1])))
        return ImageMetadataProcessedCuttings.from_metadata(metadata=metadata, cuttings=cuttings)

    return _factory


def test_padding_pixels_are_black(make_cutting):
    """Padding pixels around the grid are black, not white or some other color."""
    cutting = make_cutting(0.0, color=RED)
    img = next(stitching_cuttings([cutting], StitchingConfig()))
    assert img.getpixel((0, 0)) == BLACK  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == BLACK  # bottom-right corner


def test_cuttings_appear_in_grid_order(make_cutting):
    """Cuttings fill the grid column-major (column 0 top-to-bottom, then column 1, ...)."""
    red = make_cutting(0.0, color=RED)
    green = make_cutting(1.0, color=GREEN)
    blue = make_cutting(2.0, color=BLUE)
    config = StitchingConfig(
        cuttings=CuttingsStitchingConfig(
            num_cuttings_columns=2,
            num_cuttings_rows=2,
            output_width=300,
            output_height=300,
            padding_horizontal=20,
            padding_vertical=30,
            padding_cuttings=10,
            annotation_width=20,
            annotation_gap=10,
            font_size=12,
        )
    )
    img = np.array(next(stitching_cuttings([red, green, blue], config)))

    ys_red, xs_red = np.nonzero((img == RED).all(axis=-1))
    ys_green, xs_green = np.nonzero((img == GREEN).all(axis=-1))
    ys_blue, xs_blue = np.nonzero((img == BLUE).all(axis=-1))

    # red and green share column 0 (same x range), stacked top to bottom
    assert xs_red.min() == xs_green.min()
    assert ys_green.min() > ys_red.max()

    # blue starts column 1, to the right of column 0
    assert xs_blue.min() > xs_red.max()
    assert xs_blue.min() > xs_green.max()

    # blank gap between red and green (padding_cuttings), nothing else drawn there
    gap_y = (ys_red.max() + ys_green.min()) // 2
    assert tuple(img[gap_y, xs_red.min()]) == BLACK

    # depth annotation (white text) appears next to each cutting
    assert (img[ys_red.min() : ys_red.max(), xs_red.max() :] == WHITE).all(axis=-1).any()
