"""Tests for the segment module."""

from collections.abc import Callable

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ImageMetadata
from src.segment import segment

_IMG_SIZE = (800, 1200)
_BACKGROUND_COLOR = (20, 20, 20)  # dark, unsaturated — stands in for the tray backdrop


@pytest.fixture
def make_metadata(tmp_path):
    def _factory(
        depth_start: float,
        depth_end: float,
        draw_fn: Callable[[ImageDraw.ImageDraw], None] = lambda draw: None,
        size: tuple[int, int] = _IMG_SIZE,
    ) -> ImageMetadata:
        """Creates an ImageMetadata pointing to a synthetic TIF image built by draw_fn.

        Args:
            depth_start (float): Top-of-core depth in metres, encoded into the filename.
            depth_end (float): Bottom-of-core depth in metres, encoded into the filename.
            draw_fn (Callable[[ImageDraw.ImageDraw], None]): Callback that draws onto the
                background image (e.g. to add a core or tray rectangle). Defaults to a no-op,
                producing a flat background image.
            size (tuple[int, int]): Size of the synthetic image in pixels. Defaults to _IMG_SIZE.

        Returns:
            ImageMetadata: Metadata pointing at the saved synthetic TIF image.
        """
        filename = f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        img = Image.new("RGB", size, color=_BACKGROUND_COLOR)
        draw_fn(ImageDraw.Draw(img))
        img.save(image_path)
        return ImageMetadata(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )

    return _factory


def test_segment_detects_core_bounding_box(make_metadata):
    """A bright, unsaturated core region on a dark background is detected as-is (no trimming needed)."""
    core_box = (200, 150, 600, 1100)
    metadata = make_metadata(15.0, 16.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    # downscale_factor=1.0: detection precision is under test here, not the downscale speedup
    detections = segment([metadata], config=SegmentationConfig(downscale_factor=1.0))

    assert len(detections) == 1
    assert detections[0].result.bounding_box == core_box


def test_segment_trims_saturated_tray_by_default(make_metadata):
    """The saturated wooden tray around an unsaturated core is trimmed away by default."""
    tray_box = (150, 100, 650, 1150)
    core_box = (150, 300, 650, 950)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (draw.rectangle(tray_box, fill=(180, 120, 60)), draw.rectangle(core_box, fill=(200, 200, 200))),
    )

    detections = segment([metadata], config=SegmentationConfig(downscale_factor=1.0))

    assert detections[0].result.bounding_box == core_box


def test_segment_tray_trim_threshold_is_configurable(make_metadata):
    """Raising tray_sat_threshold above the tray's saturation disables the trim."""
    tray_box = (150, 100, 650, 1150)
    core_box = (150, 300, 650, 950)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (draw.rectangle(tray_box, fill=(180, 120, 60)), draw.rectangle(core_box, fill=(200, 200, 200))),
    )

    detections = segment([metadata], config=SegmentationConfig(tray_sat_threshold=1.1, downscale_factor=1.0))

    assert detections[0].result.bounding_box == tray_box


def test_segment_skips_image_with_no_detectable_regions(make_metadata):
    """A uniform image with no foreground region is skipped instead of raising."""
    metadata = make_metadata(15.0, 16.0)  # no draw_fn — flat background only

    detections = segment([metadata])

    assert detections == []


def test_segment_continues_after_skipping_an_unsegmentable_image(make_metadata):
    """One unsegmentable image doesn't prevent the rest of the batch from being processed."""
    blank = make_metadata(15.0, 16.0)
    core_box = (200, 150, 600, 1100)
    good = make_metadata(16.0, 17.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    detections = segment([blank, good], config=SegmentationConfig(downscale_factor=1.0))

    assert len(detections) == 1
    assert detections[0].depth_start == 16.0
    assert detections[0].result.bounding_box == core_box


def test_segment_skips_blank_non_integer_image_without_dividing_by_zero(tmp_path):
    """A blank image with a non-uint8/uint16 dtype (e.g. float32) is skipped instead of producing NaNs."""
    image_path = tmp_path / "GBC-CB50_0015.00-0016.00_vd_p.TIF"
    tifffile.imwrite(image_path, np.zeros((300, 300, 3), dtype=np.float32), photometric="rgb")
    metadata = ImageMetadata(borehole_id="GBC-CB50", depth_start=15.0, depth_end=16.0, image_path=image_path)

    detections = segment([metadata])

    assert detections == []
