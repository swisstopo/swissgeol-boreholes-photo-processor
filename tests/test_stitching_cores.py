"""Tests for the stitching module."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import CoreStitchingConfig, StitchingConfig
from src.models import CoreSegmentResult, ImageMetadataCores, ImageMetadataProcessedCores, RulerSegmentResult
from src.stitching.stitching_cores import stitching_batch_cores, stitching_cores

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
        size: tuple[int, int] = (TEST_MAX_OUTPUT_PX // 2, 20),
        color: tuple[int, int, int] = (128, 128, 128),
        px_per_unit: float = 100,
    ) -> ImageMetadataProcessedCores:
        """Creates a simple ImageMetadataProcessedCores with a single solid-color crop of the specified size."""
        filename = f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        Image.new("RGB", size, color=color).save(image_path)
        metadata = ImageMetadataCores(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )
        core = CoreSegmentResult(bbox=(0.0, 0.0, float(size[0]), float(size[1])))
        ruler = RulerSegmentResult(
            bbox=(0.0, 0.0, float(size[0]), float(size[1])), px_per_unit=px_per_unit, bbox_units=[]
        )
        return ImageMetadataProcessedCores.from_metadata(metadata=metadata, core=core, tray=core, ruler=ruler)

    return _factory


def test_padding_pixels_are_black(make_processed):
    """Padding pixels around the image are black, not white or some other color."""
    core = make_processed(0.0, 1.0, color=RED)
    config = StitchingConfig(core=CoreStitchingConfig(num_cores_per_image=6))
    batches = stitching_cores([core], config)
    img = stitching_batch_cores(
        batches[0].cores,
        batches[0].shared_ruler_steps,
        batches[0].shared_borehole_id,
        batches[0].fallback_scale,
        config,
    )
    assert len(batches) == 1
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left margin, before the ruler


@pytest.mark.parametrize(
    ("core_length_cm", "expected_ruler_steps"),
    [
        (106, 100),  # rounds down to the nearest 50cm
        (101, 100),  # rounds down to the nearest 50cm
        (130, 150),  # rounds up to the nearest 50cm
        (10, 50),  # never rounds down to 0
    ],
)
def test_ruler_length_rounds_to_nearest_50cm(make_processed, core_length_cm, expected_ruler_steps):
    """The shared ruler length is the longest core's length, rounded to the nearest 50cm."""
    core = make_processed(0.0, 1.0, size=(core_length_cm, 20), px_per_unit=1)
    config = StitchingConfig(core=CoreStitchingConfig())
    batches = stitching_cores([core], config)
    assert batches[0].shared_ruler_steps == expected_ruler_steps


def test_cores_appear_in_order_left_to_right(make_processed):
    """Cores appear in the output in the same order as the input list, from left to right."""
    # px_per_unit=1 keeps the core length (and thus the rounded-to-50cm ruler length) an exact
    # multiple of 50, so rounding doesn't perturb the expected scale below.
    red = make_processed(0.0, 1.0, color=RED, px_per_unit=1)
    green = make_processed(1.0, 2.0, color=GREEN, px_per_unit=1)
    blue = make_processed(2.0, 3.0, color=BLUE, px_per_unit=1)
    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000))
    batches = stitching_cores([red, green, blue], config)
    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
        )
    )

    assert len(batches) == 1
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
    """An outlier core is scaled to exactly fill max_core_height; shorter cores share its scale, so end up smaller."""
    # px_per_unit=1 keeps the core lengths (and thus the rounded-to-50cm ruler length) exact
    # multiples of 50, so rounding doesn't perturb the expected scale below.
    normal_a = make_processed(0.0, 1.0, size=(100, 20), color=RED, px_per_unit=1)
    normal_b = make_processed(1.0, 2.0, size=(100, 20), color=GREEN, px_per_unit=1)
    outlier = make_processed(2.0, 102.0, size=(1000, 20), color=BLUE, px_per_unit=1)

    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000))
    batches = stitching_cores([normal_a, normal_b, outlier], config)
    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
        )
    )

    assert len(batches) == 1
    ys_normal_a, _ = np.nonzero((img == RED).all(axis=-1))
    ys_normal_b, _ = np.nonzero((img == GREEN).all(axis=-1))
    ys_outlier, _ = np.nonzero((img == BLUE).all(axis=-1))

    # normal cores are 10x shorter than the outlier, so at a shared scale they're 10x shorter on screen too
    assert ys_normal_a.max() - ys_normal_a.min() + 1 == TEST_MAX_OUTPUT_PX // 10
    assert ys_normal_b.max() - ys_normal_b.min() + 1 == TEST_MAX_OUTPUT_PX // 10
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
    config = StitchingConfig(core=CoreStitchingConfig(num_cores_per_image=len(cores) - 1))
    batches = stitching_cores(cores, config)

    for idx, img in enumerate(
        [
            stitching_batch_cores(
                batch.cores,
                batch.shared_ruler_steps,
                batch.shared_borehole_id,
                batch.fallback_scale,
                config,
            )
            for batch in batches
        ]
    ):
        out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
        img.save(out_path)
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
