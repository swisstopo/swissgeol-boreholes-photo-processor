"""Tests for the segment_cuttings module."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ImageMetadataCuttings, PaperDetectionStatus
from src.segment.config import SegmentationCuttingsConfig
from src.segment.segment_cuttings import segment_black_circle, segment_cuttings, segment_pebble


@pytest.fixture
def make_metadata(tmp_path):
    def _factory(
        depth: float,
        draw_fn: Callable[[ImageDraw.ImageDraw], None] = lambda draw: None,
        size: tuple[int, int] = (400, 300),
    ) -> ImageMetadataCuttings:
        """Creates an ImageMetadataCuttings pointing to a synthetic image built by draw_fn.

        Args:
            depth (float): Depth in metres, stored on the returned metadata.
            draw_fn (Callable[[ImageDraw.ImageDraw], None]): Callback that draws onto a
                black background image (e.g. to add a circle or paper rectangle). Defaults
                to a no-op, producing a flat black image.
            size (tuple[int, int]): Size (width, height) of the synthetic image. Defaults to (400, 300).

        Returns:
            ImageMetadataCuttings: Metadata pointing at the saved synthetic image.
        """
        image_path = tmp_path / f"{depth:g}m_{id(draw_fn)}.jpg"
        img = Image.new("RGB", size, color=(0, 0, 0))
        draw_fn(ImageDraw.Draw(img))
        img.save(image_path)
        return ImageMetadataCuttings(image_path=image_path, borehole_id="B", depth=depth)

    return _factory


def test_segment_black_circle_detects_bbox_inside_circle(make_metadata):
    """A light circle on a black background yields a square bbox centered on it."""
    cx, cy, r = 200, 200, 150
    metadata = make_metadata(
        1.0, lambda draw: draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(200, 200, 200)), size=(400, 400)
    )

    result = segment_black_circle(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    x0, y0, x1, y1 = result.bbox
    assert (x1 - x0) == pytest.approx(y1 - y0)  # square
    assert (x0 + x1) / 2 == pytest.approx(cx, abs=2)
    assert (y0 + y1) / 2 == pytest.approx(cy, abs=2)


def _fake_paper(bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    """A stand-in for the regionprops object detect_paper/confirm_stripe would return."""
    return SimpleNamespace(bbox=bbox)


@pytest.fixture
def pebble_metadata(make_metadata):
    return make_metadata(1.0, size=(400, 200))  # w=400, h=200, landscape so load_image won't rotate it


def test_segment_pebble_crops_left_of_confirmed_paper(pebble_metadata):
    """A confirmed paper region crops everything left of its left edge."""
    paper = _fake_paper((20, 300, 200, 400))  # bbox = (min_row, min_col, max_row, max_col)

    with (
        patch("src.segment.segment_cuttings.detect_paper", return_value=paper),
        patch("src.segment.segment_cuttings.confirm_stripe", return_value=paper),
    ):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 300, 200)
    assert result.paper_status == PaperDetectionStatus.FOUND


def test_segment_pebble_falls_back_when_no_stripe_pattern(pebble_metadata):
    """A bright region shaped like paper but with no printed ticks nearby is rejected (issue #69)."""
    paper = _fake_paper((20, 300, 200, 400))

    with (
        patch("src.segment.segment_cuttings.detect_paper", return_value=paper),
        patch("src.segment.segment_cuttings.confirm_stripe", return_value=None),
    ):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 200)
    assert result.paper_status == PaperDetectionStatus.NO_STRIPE_PATTERN


def test_segment_pebble_falls_back_when_no_candidate_found(pebble_metadata):
    """No bright/colorless region at all keeps the full image instead of guessing."""
    with (
        patch("src.segment.segment_cuttings.detect_paper", return_value=None),
        patch("src.segment.segment_cuttings.confirm_stripe", return_value=None),
    ):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 200)
    assert result.paper_status == PaperDetectionStatus.NO_CANDIDATE


@pytest.mark.parametrize(
    ("paper_bbox", "expected_status"),
    [
        ((20, 0, 200, 100), PaperDetectionStatus.DEGENERATE_LEFT_EDGE),  # left edge at column 0
        ((20, 50, 200, 400), PaperDetectionStatus.CROPPED_TOO_MUCH),  # would crop away most of the image
    ],
)
def test_segment_pebble_rejects_unreliable_paper_region(pebble_metadata, paper_bbox, expected_status):
    """A confirmed paper region that's still geometrically implausible falls back to the full image."""
    paper = _fake_paper(paper_bbox)

    with (
        patch("src.segment.segment_cuttings.detect_paper", return_value=paper),
        patch("src.segment.segment_cuttings.confirm_stripe", return_value=paper),
    ):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 200)
    assert result.paper_status == expected_status


def test_segment_cuttings_raises_for_unknown_cut_type(make_metadata):
    """An unrecognized cut_type fails fast instead of silently doing nothing."""
    metadata = make_metadata(1.0)

    with pytest.raises(ValueError, match="Unknown cuttings type"):
        segment_cuttings([metadata], cut_type="bogus")


def test_segment_cuttings_skips_unsegmentable_image_without_crashing_batch(make_metadata):
    """A blank image with no detectable region is skipped, and the rest of the batch still runs."""
    blank = make_metadata(1.0)  # no draw_fn — flat black image, nothing to detect
    good = make_metadata(2.0, lambda draw: draw.ellipse((50, 50, 150, 150), fill=(200, 200, 200)))

    detections = segment_cuttings(
        [blank, good],
        config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
        cut_type="black_circle",
    )

    assert [d.depth for d in detections] == [2.0]
