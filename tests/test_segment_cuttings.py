"""Tests for the segment_cuttings module."""

from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ImageMetadataCuttings
from src.segment.config import SegmentationCuttingsConfig
from src.segment.segment_cuttings import segment_black_circle, segment_cuttings, segment_pebble, segment_tray


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


def test_segment_pebble_detects_region_above_paper(make_metadata):
    """Everything above the bright, unsaturated paper sheet is returned as the pebble region."""
    size = (400, 300)
    paper_top = 200
    metadata = make_metadata(
        1.0,
        lambda draw: (
            draw.rectangle((0, 0, size[0] - 1, size[1] - 1), fill=(50, 50, 150)),  # saturated, dark background
            draw.rectangle((0, paper_top, size[0] - 1, size[1] - 1), fill=(240, 240, 240)),  # bright, colorless paper
        ),
        size=size,
    )

    result = segment_pebble(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, size[0], paper_top)


def test_segment_tray_detects_textured_region(tmp_path):
    """A textured (noisy) patch on a flat background is picked out as the tray cuttings region."""
    size = (600, 400)  # width, height
    box = (150, 100, 450, 300)  # x0, y0, x1, y1

    rng = np.random.default_rng(0)
    pixels = np.full((size[1], size[0], 3), 200, dtype=np.uint8)  # flat gray background
    pixels[box[1] : box[3], box[0] : box[2]] = rng.integers(0, 255, (box[3] - box[1], box[2] - box[0], 3))
    image_path = tmp_path / "tray.jpg"
    Image.fromarray(pixels).save(image_path)
    metadata = ImageMetadataCuttings(image_path=image_path, borehole_id="B", depth=1.0)

    result = segment_tray(metadata, SegmentationCuttingsConfig(downscale_factor=1.0, tray_square=False))

    x0, y0, x1, y1 = result.bbox
    assert x0 == pytest.approx(box[0], abs=size[0] * 0.05)
    assert y0 == pytest.approx(box[1], abs=size[1] * 0.05)
    assert x1 == pytest.approx(box[2], abs=size[0] * 0.05)
    assert y1 == pytest.approx(box[3], abs=size[1] * 0.05)


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
