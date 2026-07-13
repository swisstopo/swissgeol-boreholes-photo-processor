"""Tests for the stitching module."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.stitching.stitching import stitching

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

TEST_MAX_OUTPUT_PX = 1000


@pytest.fixture
def make_processed(tmp_path):
    def _factory(
        depth_start: float,
        depth_end: float,
        size: tuple[int, int] = (20, TEST_MAX_OUTPUT_PX // 2),
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> ImageMetadataProcessed:
        """Creates a simple ImageMetadataProcessed with a single solid-color crop of the specified size."""
        filename = f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        Image.new("RGB", size, color=color).save(image_path)
        metadata = ImageMetadata(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )
        result = CoreSegmentResult(bounding_box=(0.0, 0.0, float(size[0]), float(size[1])))
        return ImageMetadataProcessed.from_metadata(metadata=metadata, result=result)

    return _factory


def test_padding_pixels_are_black(make_processed):
    """Padding pixels around the image are black, not white or some other color."""
    core = make_processed(0.0, 1.0, color=RED)
    img = next(stitching([core], num_cores_per_image=6))
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left margin, before the ruler


def test_cores_appear_in_order_left_to_right(make_processed):
    """Cores appear in the output in the same order as the input list, from left to right."""
    red = make_processed(0.0, 1.0, color=RED)
    green = make_processed(1.0, 2.0, color=GREEN)
    blue = make_processed(2.0, 3.0, color=BLUE)
    img = np.array(next(stitching([red, green, blue], core_height_px=TEST_MAX_OUTPUT_PX)))

    ys_red, xs_red = np.nonzero((img == RED).all(axis=-1))
    ys_green, xs_green = np.nonzero((img == GREEN).all(axis=-1))
    ys_blue, xs_blue = np.nonzero((img == BLUE).all(axis=-1))

    # Cores ordered
    assert xs_green.min() > xs_red.max()
    assert xs_blue.min() > xs_green.max()
    assert xs_blue.min() > xs_red.max()

    # Cores rescaled
    assert ys_red.max() - ys_red.min() + 1 == TEST_MAX_OUTPUT_PX
    assert ys_green.max() - ys_green.min() + 1 == TEST_MAX_OUTPUT_PX
    assert ys_blue.max() - ys_blue.min() + 1 == TEST_MAX_OUTPUT_PX

    # Blank spaces in between
    assert (img[ys_green.min() : ys_green.max(), (xs_green.min() + xs_red.max()) // 2] == BLACK).all()
    assert (img[ys_blue.min() : ys_blue.max(), (xs_blue.min() + xs_green.max()) // 2] == BLACK).all()

    # Top / bottom aligned
    assert ys_red.min() == ys_green.min() == ys_blue.min()
    assert ys_red.max() == ys_green.max() == ys_blue.max()

    # Depth labels below / above
    assert (img[: ys_red.min(), xs_red.min() : xs_red.max()] == WHITE).all(axis=-1).any()
    assert (img[: ys_green.min(), xs_green.min() : xs_green.max()] == WHITE).all(axis=-1).any()
    assert (img[: ys_blue.min(), xs_blue.min() : xs_blue.max()] == WHITE).all(axis=-1).any()

    # Ruler on the sides
    assert (img[ys_red.min() : ys_red.max(), : xs_red.min()] == WHITE).all(axis=-1).any()
    assert (img[ys_blue.min() : ys_blue.max(), xs_blue.max() :] == WHITE).all(axis=-1).any()


def test_outlier_core_width_matches_the_reference_core(make_processed):
    """An outlier core (see _resize_cores) is placed and sized consistently with the resize step."""
    normal_a = make_processed(0.0, 1.0, size=(20, 100), color=RED)
    normal_b = make_processed(1.0, 2.0, size=(20, 100), color=GREEN)
    outlier = make_processed(2.0, 102.0, size=(20, 1000), color=BLUE)

    img = np.array(next(stitching([normal_a, normal_b, outlier], core_height_px=TEST_MAX_OUTPUT_PX)))

    ys_normal_a, _ = np.nonzero((img == RED).all(axis=-1))
    ys_normal_b, _ = np.nonzero((img == GREEN).all(axis=-1))
    ys_outlier, _ = np.nonzero((img == BLUE).all(axis=-1))

    assert ys_normal_a.max() - ys_normal_a.min() + 1 == TEST_MAX_OUTPUT_PX
    assert ys_normal_b.max() - ys_normal_b.min() + 1 == TEST_MAX_OUTPUT_PX
    assert ys_outlier.max() - ys_outlier.min() + 1 == TEST_MAX_OUTPUT_PX


OUTPUT_DIR = Path(__file__).parent / "output" / "stitching"

_CORE_COLORS = [
    (220, 50, 50),  # red
    (220, 140, 50),  # orange
    (200, 200, 50),  # yellow
    (50, 180, 50),  # green
    (50, 140, 220),  # blue
    (140, 50, 220),  # purple
    (180, 180, 180),  # grey  — overflow core on second image
]


# @pytest.mark.skip(reason="visual inspection only — run manually")
def test_save_two_output_images(make_processed):
    """Creates two output images with 6 cores in the first and 1 core in the second, for visual inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cores = [make_processed(float(i), float(i + 1), color=_CORE_COLORS[i]) for i in range(7)]
    results = stitching(cores, num_cores_per_image=len(cores) - 1)
    for idx, img in enumerate(results):
        out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
        img.save(out_path)
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
