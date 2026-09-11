"""Tests for the stitching module."""

import numpy as np
import pytest
from PIL import Image

from src.config import CoreStitchingConfig, StitchingConfig
from src.models import CoreSegmentResult, ImageMetadataCores, ImageMetadataProcessedCores, RulerSegmentResult
from src.stitching.stitching_cores import _rounded_ruler_display_steps, stitching_batch_cores, stitching_cores

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
    config = StitchingConfig(core=CoreStitchingConfig())
    batches = stitching_cores([core], config)
    img = stitching_batch_cores(
        batches[0].cores,
        batches[0].shared_ruler_steps,
        batches[0].shared_borehole_id,
        batches[0].fallback_scale,
        config,
        is_last=batches[0].is_last,
    )
    assert len(batches) == 1
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left margin, before the ruler


@pytest.mark.parametrize(
    ("shared_ruler_steps", "expected_display_steps"),
    [
        (100, 100),  # exact match, no rounding needed
        (106, 100),  # rounds down to the nearest 50cm
        (101, 100),  # rounds down to the nearest 50cm
        (130, 150),  # rounds up to the nearest 50cm
        (10, 50),  # never rounds down to 0
    ],
)
def test_rounded_ruler_display_steps(shared_ruler_steps, expected_display_steps):
    """The ruler's displayed span rounds to the nearest 50cm, independent of the actual scale."""
    assert _rounded_ruler_display_steps(shared_ruler_steps) == expected_display_steps


def test_ruler_stops_before_a_core_longer_than_its_rounded_display_length(make_processed):
    """A core longer than the ruler's rounded-to-50cm display length still renders at full true scale.

    The ruler itself stops short of the core's bottom edge rather than being stretched to always
    cover it, since stretching it would misalign its ticks against the core's true depths.
    """
    core = make_processed(0.0, 1.0, size=(106, 20), color=RED, px_per_unit=1)
    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000))
    batches = stitching_cores([core], config)
    assert batches[0].shared_ruler_steps == 106  # true length, unrounded

    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
            is_last=batches[0].is_last,
        )
    )
    ys_red, xs_red = np.nonzero((img == RED).all(axis=-1))
    # the core itself renders at its full, undistorted true length
    assert ys_red.max() - ys_red.min() + 1 == 1000

    # but the ruler (rounded down to 100/106 of that height) stops short of the core's bottom
    near_bottom_row = ys_red.max() - 5
    ruler_region = img[near_bottom_row, : xs_red.min()]
    assert not (ruler_region == WHITE).all(axis=-1).any()


def test_cores_appear_in_order_left_to_right(make_processed):
    """Cores appear in the output in the same order as the input list, from left to right."""
    red = make_processed(0.0, 1.0, color=RED)
    green = make_processed(1.0, 2.0, color=GREEN)
    blue = make_processed(2.0, 3.0, color=BLUE)
    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000))
    batches = stitching_cores([red, green, blue], config)
    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
            is_last=batches[0].is_last,
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
    normal_a = make_processed(0.0, 1.0, size=(100, 20), color=RED)
    normal_b = make_processed(1.0, 2.0, size=(100, 20), color=GREEN)
    outlier = make_processed(2.0, 102.0, size=(1000, 20), color=BLUE)

    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000))
    batches = stitching_cores([normal_a, normal_b, outlier], config)
    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
            is_last=batches[0].is_last,
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


def test_min_core_gap_limits_cores_per_page(make_processed):
    """core_area_width and min_core_gap decide how many cores fit per page, not a fixed count."""
    # Each identical core renders at a predicted width of 40px at these default settings.
    cores = [make_processed(float(i), float(i + 1), size=(5000, 20)) for i in range(7)]
    # 6 cores * 40px + 5 gaps * 40px == 440, so a 7th core has to overflow onto a second page.
    config = StitchingConfig(core=CoreStitchingConfig(core_area_width=440, min_core_gap=40))
    batches = stitching_cores(cores, config)

    assert len(batches) == 2
    assert len(batches[0].cores) == 6
    assert len(batches[1].cores) == 1
    assert batches[0].is_last is False
    assert batches[1].is_last is True


def test_last_page_packs_tight_and_left_aligned(make_processed):
    """The last page's cores are packed at exactly min_core_gap, leaving core_area_width's leftover black."""
    red = make_processed(0.0, 1.0, color=RED, px_per_unit=1)
    green = make_processed(1.0, 2.0, color=GREEN, px_per_unit=1)
    config = StitchingConfig(core=CoreStitchingConfig(max_core_height=1000, core_area_width=1000, min_core_gap=10))
    batches = stitching_cores([red, green], config)
    assert len(batches) == 1
    assert batches[0].is_last is True

    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
            is_last=batches[0].is_last,
        )
    )

    ys_red, xs_red = np.nonzero((img == RED).all(axis=-1))
    ys_green, xs_green = np.nonzero((img == GREEN).all(axis=-1))

    core_width = 40
    assert xs_red.max() - xs_red.min() + 1 == core_width
    assert xs_green.max() - xs_green.min() + 1 == core_width

    # Gap is exactly min_core_gap, not stretched to fill core_area_width.
    gap = xs_green.min() - xs_red.max() - 1
    assert gap == config.core.min_core_gap

    # Leftover core_area_width stays black instead of being spread between the cores.
    right_of_cores = img[ys_green.min() : ys_green.max(), xs_green.max() + 1 : xs_green.max() + 1 + 100]
    assert (right_of_cores == BLACK).all()


def test_non_last_page_spreads_cores_to_fill_core_area_width(make_processed):
    """A full (non-last) page still spreads its leftover width evenly between cores, flush against both rulers."""
    # With 40px-per-core and core_area_width=300, min_core_gap=40, 4 cores fit (4*40+3*40==280)
    # but a 5th wouldn't (+80==360>300), leaving this non-last page with leftover width to spread.
    colors = [RED, GREEN, BLUE, (200, 200, 0)]
    cores = [make_processed(float(i), float(i + 1), size=(5000, 20), color=colors[i]) for i in range(4)]
    cores += [make_processed(float(i), float(i + 1), size=(5000, 20)) for i in range(4, 7)]
    config = StitchingConfig(core=CoreStitchingConfig(core_area_width=300, min_core_gap=40))
    batches = stitching_cores(cores, config)

    assert len(batches) == 2
    assert len(batches[0].cores) == 4
    assert batches[0].is_last is False

    img = np.array(
        stitching_batch_cores(
            batches[0].cores,
            batches[0].shared_ruler_steps,
            batches[0].shared_borehole_id,
            batches[0].fallback_scale,
            config,
            is_last=batches[0].is_last,
        )
    )

    xs_by_color = [np.nonzero((img == color).all(axis=-1))[1] for color in colors]
    gaps = [xs_by_color[i + 1].min() - xs_by_color[i].max() - 1 for i in range(3)]

    # Gap is spread wider than min_core_gap, and equal between every pair of cores.
    assert all(gap > config.core.min_core_gap for gap in gaps)
    assert len(set(gaps)) == 1
