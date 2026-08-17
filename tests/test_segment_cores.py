"""Tests for the segment module."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw

from src.config import (
    SegmentationConfig,
    SegmentationCoreConfig,
    SegmentationCoreTrimConfig,
    SegmentationRulerConfig,
    SegmentationTrayGroupConfig,
    SegmentationTraySingleConfig,
)
from src.models import ImageMetadataCores, ImageSegmentResult, RulerSegmentResult, TraySegmentResult
from src.segment.segment_cores import segment_cores
from src.segment.utils.core import segment_core
from src.segment.utils.misc import group_images_by_shape
from src.segment.utils.ruler import ProcessRulerGroupByShape
from src.segment.utils.tray import ProcessTrayGroupByShape, _bbox_skimage_intersection

_IMG_SIZE = (1200, 800)
_BACKGROUND_COLOR = (20, 20, 20)  # dark, unsaturated — stands in for the tray backdrop


@pytest.fixture
def make_metadata(tmp_path):
    def _factory(
        depth_start: float,
        depth_end: float,
        draw_fn: Callable[[ImageDraw.ImageDraw], None] = lambda draw: None,
        size: tuple[int, int] = _IMG_SIZE,
    ) -> ImageMetadataCores:
        """Creates an ImageMetadataCores pointing to a synthetic TIF image built by draw_fn.

        Args:
            depth_start (float): Top-of-core depth in metres, encoded into the filename.
            depth_end (float): Bottom-of-core depth in metres, encoded into the filename.
            draw_fn (Callable[[ImageDraw.ImageDraw], None]): Callback that draws onto the
                background image (e.g. to add a core or tray rectangle). Defaults to a no-op,
                producing a flat background image.
            size (tuple[int, int]): Size of the synthetic image in pixels. Defaults to _IMG_SIZE.

        Returns:
            ImageMetadataCores: Metadata pointing at the saved synthetic TIF image.
        """
        filename = f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        img = Image.new("RGB", size, color=_BACKGROUND_COLOR)
        draw_fn(ImageDraw.Draw(img))
        img.save(image_path)
        return ImageMetadataCores(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )

    return _factory


@pytest.fixture
def example(tmp_path) -> ImageMetadataCores:
    """Copies the real example core/tray/ruler photo into tmp_path as a .tif and wraps it in ImageMetadataCores."""
    path_source_ = Path("examples/EX-EX_0001.00-002.00.jpg")
    path_dest = tmp_path / (path_source_.stem + ".tif")
    Image.open(path_source_).save(path_dest)
    return ImageMetadataCores.from_path(path_dest)


def test_segment_example(example):
    """End-to-end test of segment_cores() against the real example photo, checking metadata and bbox geometry."""
    detections = segment_cores(
        [example],
        config=SegmentationConfig(core=SegmentationCoreConfig(ruler=SegmentationRulerConfig(downscale_factor=1.0))),
    )

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
    assert pytest.approx(detections[0].ruler.px_per_unit, rel=0.02) == 100


def test_segment_detects_core_bbox(make_metadata):
    """A bright, unsaturated core region on a dark background is detected as-is (no trimming needed)."""
    core_box = (200, 150, 600, 700)
    metadata = make_metadata(15.0, 16.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    # downscale_factor=1.0: detection precision is under test here, not the downscale speedup
    detections = segment_cores(
        [metadata],
        config=SegmentationConfig(
            core=SegmentationCoreConfig(
                tray_single=SegmentationTraySingleConfig(downscale_factor=1),
                core=SegmentationCoreTrimConfig(downscale_factor=1),
            ),
        ),
    )

    assert len(detections) == 1
    assert detections[0].core is not None
    assert detections[0].core.bbox == core_box


def test_segment_wood_tray_trim_threshold_is_configurable(make_metadata):
    """Wood tray saturation trimming is configurable."""
    tray_box = (150, 100, 1050, 700)
    core_box = (150, 250, 1050, 550)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (draw.rectangle(tray_box, fill=(180, 120, 60)), draw.rectangle(core_box, fill=(200, 200, 200))),
    )

    # Default (correct threshold)
    detection_core = segment_core(
        metadata,
        tray=ImageSegmentResult(bbox=tray_box),
        config=SegmentationCoreTrimConfig(downscale_factor=1),
    )

    # Badly configured threshold
    detection_core_out = segment_core(
        metadata,
        tray=ImageSegmentResult(bbox=tray_box),
        config=SegmentationCoreTrimConfig(downscale_factor=1, wood_sat_threshold=1.1),
    )

    assert detection_core is not None
    assert detection_core_out is not None
    assert detection_core.bbox == core_box
    assert detection_core_out.bbox == tray_box


def test_segment_core_trims_black_background_left_right(make_metadata):
    """Black background left/right of a full-height, unsaturated core is trimmed via the value channel alone."""
    size = (400, 100)
    core_box = (100, 0, 300, 99)
    metadata = make_metadata(15.0, 16.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)), size=size)
    tray = TraySegmentResult(bbox=(0, 0, size[0] - 1, size[1] - 1))

    result = segment_core(metadata, tray, config=SegmentationCoreTrimConfig(downscale_factor=1))

    assert result is not None
    assert result.bbox == core_box


def test_segment_core_splits_fragmented_core_into_segments(make_metadata):
    """A core split in two by a black gap yields two bbox_segments and a bbox spanning their union."""
    size = (400, 100)
    left_box = (50, 0, 150, 99)
    right_box = (250, 0, 350, 99)
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (
            draw.rectangle(left_box, fill=(200, 200, 200)),
            draw.rectangle(right_box, fill=(200, 200, 200)),
        ),
        size=size,
    )
    tray = TraySegmentResult(bbox=(0, 0, size[0] - 1, size[1] - 1))

    result = segment_core(metadata, tray, config=SegmentationCoreTrimConfig(downscale_factor=1))

    assert result is not None
    assert result.bbox == (left_box[0], 0, right_box[2], 99)
    assert result.bbox_segments is not None
    assert len(result.bbox_segments) == 2
    assert sorted(result.bbox_segments) == sorted([left_box, right_box])


def test_segment_core_drops_thin_segments(make_metadata):
    """A segment thinner than min_segment_height_px is dropped and doesn't widen the core bbox."""
    size = (400, 100)
    core_box = (100, 0, 300, 99)
    segment_box = (10, 0, 14, 99)  # 5px wide, well under the default min_segment_height_px of 10
    metadata = make_metadata(
        15.0,
        16.0,
        lambda draw: (
            draw.rectangle(core_box, fill=(200, 200, 200)),
            draw.rectangle(segment_box, fill=(200, 200, 200)),
        ),
        size=size,
    )
    tray = TraySegmentResult(bbox=(0, 0, size[0] - 1, size[1] - 1))

    result = segment_core(metadata, tray, config=SegmentationCoreTrimConfig(downscale_factor=1))

    assert result is not None
    assert result.bbox == core_box  # segment excluded, doesn't pull the left edge out to x=10
    assert result.bbox_segments is not None
    assert len(result.bbox_segments) == 1


def test_segment_skips_image_with_no_detectable_regions(make_metadata):
    """A uniform image with no foreground region is skipped instead of raising."""
    metadata = make_metadata(15.0, 16.0)  # no draw_fn — flat background only

    detections = segment_cores(
        [metadata],
        config=SegmentationConfig(
            core=SegmentationCoreConfig(
                tray_single=SegmentationTraySingleConfig(downscale_factor=1),
                core=SegmentationCoreTrimConfig(downscale_factor=1),
            ),
        ),
    )

    assert len(detections) == 1
    assert detections[0].tray is not None
    assert detections[0].tray.bbox == (0, 0, _IMG_SIZE[0] - 1, _IMG_SIZE[1] - 1)
    assert detections[0].core is not None
    assert detections[0].core.bbox == (0, 0, _IMG_SIZE[0] - 1, _IMG_SIZE[1] - 1)


def test_segment_continues_after_skipping_an_unsegmentable_image(make_metadata):
    """One unsegmentable image doesn't prevent the rest of the batch from being processed."""
    blank = make_metadata(15.0, 16.0)
    core_box = (200, 150, 600, 700)
    good = make_metadata(16.0, 17.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    detections = segment_cores(
        [blank, good],
        config=SegmentationConfig(
            core=SegmentationCoreConfig(
                tray_single=SegmentationTraySingleConfig(downscale_factor=1),
                core=SegmentationCoreTrimConfig(downscale_factor=1),
            ),
        ),
    )

    assert len(detections) == 2
    assert detections[0].depth_start == 15.0
    assert detections[0].core is not None
    assert detections[0].core.bbox == (0, 0, _IMG_SIZE[0] - 1, _IMG_SIZE[1] - 1)

    assert detections[1].depth_start == 16.0
    assert detections[1].core is not None
    assert detections[1].core.bbox == core_box


def test_segment_skips_blank_non_integer_image_without_crashing_batch(tmp_path, make_metadata):
    """A blank image with a non-uint8/uint16 dtype (e.g. float32) is skipped, not crashing the whole batch."""
    bad_image_path = tmp_path / "GBC-CB50_0015.00-0016.00_vd_p.TIF"
    tifffile.imwrite(bad_image_path, np.zeros((300, 300, 3), dtype=np.float32), photometric="rgb")
    bad = ImageMetadataCores(borehole_id="GBC-CB50", depth_start=15.0, depth_end=16.0, image_path=bad_image_path)

    core_box = (200, 150, 600, 700)
    good = make_metadata(16.0, 17.0, lambda draw: draw.rectangle(core_box, fill=(200, 200, 200)))

    detections = segment_cores(
        [bad, good],
        config=SegmentationConfig(
            core=SegmentationCoreConfig(
                tray_single=SegmentationTraySingleConfig(downscale_factor=1),
                core=SegmentationCoreTrimConfig(downscale_factor=1),
            ),
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
        ProcessTrayGroupByShape(
            SegmentationTrayGroupConfig(downscale_factor=1, n_min_foreground=10),
        ).run(imgs)
        == {}
    )


def test_estimate_foreground_skips_shape_group_below_n_min(make_metadata):
    """A shape group with too few images is dropped, but a sufficient one still succeeds."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(100, 100)) for i in range(10)]
    imgs.append(make_metadata(25.0, 26.0, size=(50, 50)))

    results = ProcessTrayGroupByShape(
        SegmentationTrayGroupConfig(downscale_factor=1, n_min_foreground=10),
    ).run(imgs)

    assert set(results.keys()) == {(100, 100, 3)}


def test_estimate_foreground_highlights_the_varying_core_region(make_metadata):
    """Pixels that change across the batch (the core) get a higher std than the static background."""
    core_box = (0, 50, 100, 100)
    fills = [50, 90, 130, 170, 210, 250]
    imgs = [
        make_metadata(15.0 + i, 16.0 + i, lambda draw, f=f: draw.rectangle(core_box, fill=(f, f, f)), size=(100, 100))
        for i, f in enumerate(fills)
    ]

    results = ProcessTrayGroupByShape(
        SegmentationTrayGroupConfig(downscale_factor=1, n_min_foreground=5),
    ).run(imgs)

    assert results is not None
    assert set(results.keys()) == {(100, 100, 3)}

    x_min, y_min, x_max, _ = results[(100, 100, 3)].bbox
    assert int(x_max) - int(x_min) + 1 == 100  # Spans left to right
    assert int(y_min) > 50  # Bottom half is moving, not upper


def test_group_images_by_shape_groups_by_dimensions(make_metadata):
    """Images are grouped by their (height, width, channels) shape, independent of depth/order."""
    small = [make_metadata(15.0 + i, 16.0 + i, size=(50, 50)) for i in range(2)]
    large = [make_metadata(30.0 + i, 31.0 + i, size=(80, 80)) for i in range(3)]

    grouped = group_images_by_shape(small + large)

    assert set(grouped) == {small[0].shape, large[0].shape}
    assert grouped[small[0].shape] == small
    assert grouped[large[0].shape] == large


def test_process_tray_group_by_shape_estimates_independently_per_shape(make_metadata):
    """Each image-shape group gets its own independently estimated shared-foreground bbox."""
    fills = [50, 90, 130, 170, 210]

    def make_group(size, depth_offset):
        core_box = (0, size[1] // 2, size[0], size[1])  # bottom half varies
        imgs = [
            make_metadata(
                depth_offset + i,
                depth_offset + i + 1,
                lambda draw, f=f: draw.rectangle(core_box, fill=(f, f, f)),
                size=size,
            )
            for i, f in enumerate(fills)
        ]
        return imgs, core_box

    group_a, core_box_a = make_group((100, 100), 15.0)
    group_b, core_box_b = make_group((60, 60), 30.0)

    results = ProcessTrayGroupByShape(
        SegmentationTrayGroupConfig(downscale_factor=1, n_min_foreground=5),
    ).run(group_a + group_b)

    shape_a, shape_b = group_a[0].shape, group_b[0].shape
    assert set(results) == {shape_a, shape_b}

    for core_box, shape in [(core_box_a, shape_a), (core_box_b, shape_b)]:
        x_min, y_min, x_max, _ = results[shape].bbox
        assert int(x_max) - int(x_min) + 1 == core_box[2] - core_box[0]  # spans full width
        assert int(y_min) > core_box[1] - 1  # bottom half is moving, not upper


def test_process_ruler_group_by_shape_skips_shape_group_below_n_min(make_metadata):
    """A shape group smaller than n_min_ruler is skipped, not OCR'd and aggregated."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(50, 50)) for i in range(3)]

    results = ProcessRulerGroupByShape(config=SegmentationRulerConfig(downscale_factor=1, n_min_ruler=5)).run(imgs)

    assert results == {}


def test_process_ruler_group_by_shape_picks_median_scale_detection():
    """Among several detections, the one with the median px_per_unit is kept, not the first or last."""
    detections = [
        RulerSegmentResult(bbox=(0, 0, 10, 10), px_per_unit=9.0, bbox_units=[]),
        RulerSegmentResult(bbox=(0, 0, 10, 10), px_per_unit=5.0, bbox_units=[]),
        RulerSegmentResult(bbox=(0, 0, 10, 10), px_per_unit=7.0, bbox_units=[]),
    ]

    result = ProcessRulerGroupByShape(SegmentationRulerConfig())._aggregate(detections)

    assert result is not None
    assert result.px_per_unit == 7.0  # median of 9.0, 5.0, 7.0


def test_process_ruler_group_by_shape_skips_shape_group_when_no_image_detects_a_ruler(make_metadata):
    """A shape group where no image yields a ruler detection is dropped from the results, like tray."""
    imgs = [make_metadata(15.0 + i, 16.0 + i, size=(50, 50)) for i in range(2)]

    results = ProcessRulerGroupByShape(
        SegmentationRulerConfig(downscale_factor=1, n_min_ruler=2),
    ).run(imgs)

    assert results == {}


def test_bbox_skimage_intersection():
    """Test proper image intersection for overlapping bbox."""
    assert _bbox_skimage_intersection((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert _bbox_skimage_intersection((0, 0, 10, 10), (10, 10, 20, 20)) is False
