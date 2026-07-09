"""Tests for the segment module."""

from collections.abc import Callable

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ImageMetadata
from src.segment.segment import segment
from src.segment.utils import _estimate_foreground

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


def test_estimate_foreground_returns_none_below_n_min(make_metadata):
    """Fewer than n_min successfully loaded images isn't enough for a reliable per-pixel std."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(50, 50)) for i in range(4)]

    assert _estimate_foreground(imgs, factor=1.0, n_min=5) is None


def test_estimate_foreground_returns_none_on_inconsistent_image_sizes(make_metadata):
    """A batch with a mismatched image size can't be stacked into a single std map."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(100, 100)) for i in range(10)]
    imgs.append(make_metadata(25.0, 26.0, size=(50, 50)))

    assert _estimate_foreground(imgs, factor=1.0, n_min=5) is None


def test_estimate_foreground_highlights_the_varying_core_region(make_metadata):
    """Pixels that change across the batch (the core) get a higher std than the static background."""
    core_box = (0, 50, 100, 100)
    fills = [50, 90, 130, 170, 210, 250]
    imgs = [
        make_metadata(15.0 + i, 16.0 + i, lambda draw, f=f: draw.rectangle(core_box, fill=(f, f, f)), size=(100, 100))
        for i, f in enumerate(fills)
    ]

    foreground = _estimate_foreground(imgs, factor=1.0, n_min=5)

    assert foreground is not None
    assert foreground.shape == (100, 100)
    background_std = foreground[0:25].mean()  # Top part as backgorund
    core_std = foreground[75].mean()  # Bottom part as core
    assert core_std > background_std
    assert background_std == pytest.approx(0.0, abs=1e-9)
