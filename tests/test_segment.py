"""Tests for the segment module."""

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pytest import approx

from src.config import (
    SegmentationConfig,
    SegmentationCoreConfig,
    SegmentationRulerConfig,
    SegmentationTrayMultipleConfig,
    SegmentationTraySingleConfig,
)
from src.models import ImageMetadata
from src.segment.segment import segment
from src.segment.utils import segment_tray_multiple

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


@pytest.fixture
def example(tmp_path) -> ImageMetadata:
    """Copies the real example core/tray/ruler photo into tmp_path as a .tif and wraps it in ImageMetadata."""
    path_source_ = Path("examples/EX-EX_0001.00-002.00.jpg")
    path_dest = tmp_path / (path_source_.stem + ".tif")
    Image.open(path_source_).save(path_dest)
    return ImageMetadata.from_path(path_dest)


def test_segment_example(example):
    """End-to-end test of segment() against the real example photo, checking metadata and bbox geometry."""
    detections = segment([example], config=SegmentationConfig(ruler=SegmentationRulerConfig(downscale_factor=1.0)))

    # Check metadata
    assert len(detections) == 1
    assert detections[0].borehole_id == "EX-EX"
    assert detections[0].depth_start == 1.0
    assert detections[0].depth_end == 2.0

    # Check segmentation results
    assert detections[0].core is not None
    assert detections[0].tray is not None
    assert detections[0].ruler is not None

    # Check core is contained within tray and trimmed (y axis)
    assert detections[0].core.bbox[0] >= detections[0].tray.bbox[0]
    assert detections[0].core.bbox[1] > detections[0].tray.bbox[1]
    assert detections[0].core.bbox[2] <= detections[0].tray.bbox[2]
    assert detections[0].core.bbox[3] < detections[0].tray.bbox[3]

    # Ruler detected with proper resolution (2% relative error)
    assert approx(detections[0].ruler.px_per_unit, rel=0.02) == 100


def test_segment_detects_core_bbox(make_metadata):
    """A bright, unsaturated core region on a dark background is detected as-is (no trimming needed)."""
    core_box = (200, 150, 600, 1100)
    metadata = make_metadata(15.0, 16.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    # downscale_factor=1.0: detection precision is under test here, not the downscale speedup
    detections = segment(
        [metadata],
        config=SegmentationConfig(
            tray_single=SegmentationTraySingleConfig(downscale_factor=1),
            core=SegmentationCoreConfig(downscale_factor=1),
        ),
    )

    assert len(detections) == 1
    assert detections[0].core is not None
    assert detections[0].core.bbox == core_box


def test_segment_trims_saturated_tray_by_default(make_metadata):
    """The saturated wooden tray around an unsaturated core is trimmed away by default."""
    tray_box = (150, 100, 650, 1150)
    core_box = (150, 300, 650, 950)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (draw.rectangle(tray_box, fill=(180, 120, 60)), draw.rectangle(core_box, fill=(200, 200, 200))),
    )

    detections = segment(
        [metadata],
        config=SegmentationConfig(
            tray_single=SegmentationTraySingleConfig(downscale_factor=1),
            core=SegmentationCoreConfig(downscale_factor=1),
        ),
    )

    assert detections[0].core is not None
    assert detections[0].core.bbox == core_box


def test_segment_tray_trim_threshold_is_configurable(make_metadata):
    """Raising tray_sat_threshold above the tray's saturation disables the trim."""
    tray_box = (150, 100, 650, 1150)
    core_box = (150, 300, 650, 950)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (draw.rectangle(tray_box, fill=(180, 120, 60)), draw.rectangle(core_box, fill=(200, 200, 200))),
    )

    detections = segment(
        [metadata],
        config=SegmentationConfig(
            tray_single=SegmentationTraySingleConfig(downscale_factor=1),
            core=SegmentationCoreConfig(downscale_factor=1, tray_sat_threshold=1.1),
        ),
    )

    assert detections[0].tray is not None
    assert detections[0].tray.bbox == tray_box


def test_segment_skips_image_with_no_detectable_regions(make_metadata):
    """A uniform image with no foreground region is skipped instead of raising."""
    metadata = make_metadata(15.0, 16.0)  # no draw_fn — flat background only

    detections = segment(
        [metadata],
        config=SegmentationConfig(
            tray_single=SegmentationTraySingleConfig(downscale_factor=1),
            core=SegmentationCoreConfig(downscale_factor=1),
        ),
    )

    assert detections == []


def test_segment_continues_after_skipping_an_unsegmentable_image(make_metadata):
    """One unsegmentable image doesn't prevent the rest of the batch from being processed."""
    blank = make_metadata(15.0, 16.0)
    core_box = (200, 150, 600, 1100)
    good = make_metadata(16.0, 17.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    detections = segment(
        [blank, good],
        config=SegmentationConfig(
            tray_single=SegmentationTraySingleConfig(downscale_factor=1),
            core=SegmentationCoreConfig(downscale_factor=1),
        ),
    )

    assert len(detections) == 1
    assert detections[0].depth_start == 16.0
    assert detections[0].core is not None
    assert detections[0].core.bbox == core_box


def test_estimate_foreground_returns_none_below_n_min(make_metadata):
    """Fewer than n_min successfully loaded images isn't enough for a reliable per-pixel std."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(50, 50)) for i in range(4)]

    assert (
        segment_tray_multiple(
            imgs,
            config=SegmentationTrayMultipleConfig(downscale_factor=1, n_min_foreground=10),
        )
        is None
    )


def test_estimate_foreground_returns_none_on_inconsistent_image_sizes(make_metadata):
    """A batch with a mismatched image size can't be stacked into a single std map."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(100, 100)) for i in range(10)]
    imgs.append(make_metadata(25.0, 26.0, size=(50, 50)))

    assert (
        segment_tray_multiple(
            imgs,
            config=SegmentationTrayMultipleConfig(downscale_factor=1, n_min_foreground=10),
        )
        is None
    )


def test_estimate_foreground_highlights_the_varying_core_region(make_metadata):
    """Pixels that change across the batch (the core) get a higher std than the static background."""
    core_box = (0, 50, 100, 100)
    fills = [50, 90, 130, 170, 210, 250]
    imgs = [
        make_metadata(15.0 + i, 16.0 + i, lambda draw, f=f: draw.rectangle(core_box, fill=(f, f, f)), size=(100, 100))
        for i, f in enumerate(fills)
    ]

    tray = segment_tray_multiple(
        imgs,
        config=SegmentationTrayMultipleConfig(downscale_factor=1, n_min_foreground=5),
    )
    assert tray is not None

    x_min, y_min, x_max, _ = tray.bbox
    assert int(x_max) - int(x_min) + 1 == 100  # Spans left to right
    assert int(y_min) > 50  # Bottom half is moving, not upper
