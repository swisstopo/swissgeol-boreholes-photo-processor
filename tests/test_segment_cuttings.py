"""Tests for the segment_cuttings module."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ApproachType, CuttingsSegmentResult, ImageMetadataCuttings, PaperDetectionStatus
from src.segment.config import (
    SegmentationCuttingsConfig,
    SegmentationCuttingsPebbleGroupConfig,
    SegmentationCuttingsTrayConfig,
)
from src.segment.segment_cuttings import (
    DEFAULT_CUT_TYPE,
    _guard_degenerate_bbox,
    _is_full_frame_bbox,
    _log_crop_size_consistency,
    _log_fallback_rate,
    _normalize_tray_scale,
    segment_black_circle,
    segment_cuttings,
    segment_full,
    segment_pebble,
    segment_tray,
)
from src.segment.utils.cuttings import ProcessPebblePaperGroupByShape, resolve_paper_crop


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


def test_segment_full_returns_whole_image_bbox(make_metadata):
    """No cropping: the bbox should cover the entire image."""
    metadata = make_metadata(1.0, size=(400, 300))

    result = segment_full(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 300)


def test_default_cut_type_is_full():
    """The default cuttings segmentation should take the entire image, with no cropping."""
    assert DEFAULT_CUT_TYPE == "full"


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


def _make_textured_metadata(tmp_path, depth: float, size: tuple[int, int], patches: list[tuple[int, int, int, int]]):
    """Creates an ImageMetadataCuttings for a flat image with noisy (textured) rectangular patches.

    segment_tray keys off local edge-density, so plain ImageDraw shapes (flat fill) aren't textured
    enough to register; each patch gets independent random noise instead.

    Args:
        tmp_path: pytest tmp_path fixture, used as the save location.
        depth (float): Depth in metres, stored on the returned metadata.
        size (tuple[int, int]): Size (width, height) of the image.
        patches (list[tuple[int, int, int, int]]): Noisy rectangles as (x0, y0, x1, y1).

    Returns:
        ImageMetadataCuttings: Metadata pointing at the saved synthetic image.
    """
    rng = np.random.default_rng(0)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x0, y0, x1, y1 in patches:
        img[y0:y1, x0:x1] = rng.integers(0, 255, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)

    image_path = tmp_path / f"{depth:g}m_tray.jpg"
    Image.fromarray(img).save(image_path, quality=95)
    return ImageMetadataCuttings(image_path=image_path, borehole_id="B", depth=depth)


def test_segment_tray_detects_bbox_of_textured_region(tmp_path):
    """A single textured patch on a flat background yields a bbox around that patch."""
    pile = (50, 150, 350, 380)
    metadata = _make_textured_metadata(tmp_path, 1.0, size=(500, 400), patches=[pile])

    result = segment_tray(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    x0, y0, x1, y1 = result.bbox
    assert (x0, y0) == pytest.approx(pile[:2], abs=5)
    assert (x1, y1) == pytest.approx(pile[2:], abs=5)


def test_segment_tray_excludes_disconnected_label_blob(tmp_path):
    """A separate textured label above the pile (printed text has its own edge-density) is excluded.

    Regression test: taking quantiles over every mask pixel (including the label's) would pull
    the box upward to cover the label too.
    """
    pile = (50, 150, 350, 380)
    label_tag = (100, 10, 300, 60)  # disconnected from the pile by a smooth gap
    metadata = _make_textured_metadata(tmp_path, 1.0, size=(500, 400), patches=[pile, label_tag])

    result = segment_tray(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    x0, y0, x1, y1 = result.bbox
    assert y0 > label_tag[3]  # box top starts below the label, not at/above it
    assert (x0, y0) == pytest.approx(pile[:2], abs=5)
    assert (x1, y1) == pytest.approx(pile[2:], abs=5)


def _fake_paper(bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    """A stand-in for the regionprops object detect_paper would return."""
    return SimpleNamespace(bbox=bbox)


@pytest.fixture
def pebble_metadata(make_metadata):
    return make_metadata(1.0, size=(400, 200))  # w=400, h=200, landscape so load_image won't rotate it


def test_segment_pebble_crops_left_of_confirmed_paper(pebble_metadata):
    """A detected paper region crops everything left of its left edge."""
    paper = _fake_paper((20, 300, 200, 400))  # bbox = (min_row, min_col, max_row, max_col)

    with patch("src.segment.segment_cuttings.detect_paper", return_value=paper):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 300, 200)
    assert result.paper_status == PaperDetectionStatus.FOUND


def test_segment_pebble_falls_back_when_no_candidate_found(pebble_metadata):
    """No bright/colorless region at all keeps the full image instead of guessing."""
    with patch("src.segment.segment_cuttings.detect_paper", return_value=None):
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
    """A detected paper region that's still geometrically implausible falls back to the full image."""
    paper = _fake_paper(paper_bbox)

    with patch("src.segment.segment_cuttings.detect_paper", return_value=paper):
        result = segment_pebble(pebble_metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 200)
    assert result.paper_status == expected_status


def test_segment_cuttings_raises_for_unknown_cut_type(make_metadata):
    """An unrecognized cut_type fails fast instead of silently doing nothing."""
    metadata = make_metadata(1.0)

    with pytest.raises(ValueError, match="Unknown cuttings type"):
        segment_cuttings([metadata], cut_type="bogus")


def test_segment_cuttings_logs_the_chosen_cut_type(make_metadata, caplog):
    """The chosen cut_type is always visible in run output, not just in --help's shown default."""
    metadata = make_metadata(1.0)

    with caplog.at_level("INFO"):
        segment_cuttings(
            [metadata],
            config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
            cut_type="black_circle",
        )

    assert "cut_type=black_circle" in caplog.text


def test_segment_cuttings_falls_back_to_full_image_when_nothing_detected(make_metadata):
    """A blank image with no detectable region falls back to the full image instead of being dropped."""
    blank = make_metadata(1.0, size=(400, 300))  # no draw_fn — flat black image, nothing to detect
    good = make_metadata(2.0, lambda draw: draw.ellipse((50, 50, 150, 150), fill=(200, 200, 200)))

    detections = segment_cuttings(
        [blank, good],
        config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
        cut_type="black_circle",
    )

    assert [d.depth for d in detections] == [1.0, 2.0]
    blank_detection = next(d for d in detections if d.depth == 1.0)
    assert blank_detection.cuttings is not None
    assert blank_detection.cuttings.bbox == (0, 0, 400, 300)


@pytest.mark.parametrize(
    ("paper_bbox", "expected_status", "expected_bbox"),
    [
        ((20, 300, 200, 400), PaperDetectionStatus.FOUND, (0, 0, 300, 200)),
        (None, PaperDetectionStatus.NO_CANDIDATE, (0, 0, 400, 200)),
        ((20, 0, 200, 100), PaperDetectionStatus.DEGENERATE_LEFT_EDGE, (0, 0, 400, 200)),  # left edge at column 0
        ((20, 50, 200, 400), PaperDetectionStatus.CROPPED_TOO_MUCH, (0, 0, 400, 200)),  # crops away most of the image
    ],
)
def test_resolve_paper_crop_outcomes(paper_bbox, expected_status, expected_bbox):
    """Shared decision logic used by both per-image and group paper detection."""
    paper = _fake_paper(paper_bbox) if paper_bbox is not None else None

    status, bbox = resolve_paper_crop(paper, h=200, w=400, max_cropped_frac=0.5)

    assert status == expected_status
    assert bbox == expected_bbox


_PAPER_COLOR = (235, 235, 235)  # bright, colorless — stands in for the reference paper sheet
_PAPER_BOX = (140, 30, 199, 100)  # anchored to the right edge, away from top/bottom/left


@pytest.fixture
def make_group_metadata(tmp_path):
    def _factory(n: int, fills: list[int] | None = None, draw_paper: bool = True, size=(200, 150)):
        """Creates n synthetic images of a shared shape, each with different "cuttings" content.

        A varying-fill background stands in for cuttings material (different rock every shot);
        a paper rectangle, when drawn, is identical across every image so it survives averaging.
        """
        fills = fills or [20 + 10 * i for i in range(n)]
        imgs = []
        for i in range(n):
            f = fills[i % len(fills)]
            img = Image.new("RGB", size, color=(f, f, f))
            if draw_paper:
                ImageDraw.Draw(img).rectangle(_PAPER_BOX, fill=_PAPER_COLOR)
            path = tmp_path / f"{i}m.jpg"
            img.save(path)
            imgs.append(ImageMetadataCuttings(image_path=path, borehole_id="B", depth=float(i)))
        return imgs

    return _factory


def test_process_pebble_paper_group_by_shape_skips_shape_group_below_n_min(make_group_metadata):
    """A shape group with too few images doesn't get a shared estimate."""
    imgs = make_group_metadata(4)

    results = ProcessPebblePaperGroupByShape(
        SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
    ).run(imgs)

    assert results == {}


def test_process_pebble_paper_group_by_shape_finds_consistent_paper_region(make_group_metadata):
    """A paper region that's identical across the group survives averaging and gets detected."""
    imgs = make_group_metadata(12)

    results = ProcessPebblePaperGroupByShape(
        SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
    ).run(imgs)

    shape = imgs[0].shape
    assert set(results) == {shape}
    result = results[shape]
    assert result.paper_status == PaperDetectionStatus.FOUND
    assert result.approach == ApproachType.GROUP
    assert result.bbox == (0, 0, pytest.approx(_PAPER_BOX[0], abs=6), 150)


def test_process_pebble_paper_group_by_shape_ignores_a_fixed_dark_region(make_group_metadata):
    """A region that's fixed across the group but dark (e.g. a lens-vignette corner) isn't mistaken for paper."""
    imgs = make_group_metadata(12, draw_paper=False)
    for img_metadata in imgs:
        with Image.open(img_metadata.image_path) as img:
            ImageDraw.Draw(img).rectangle(_PAPER_BOX, fill=(10, 10, 10))
            img.save(img_metadata.image_path)

    results = ProcessPebblePaperGroupByShape(
        SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
    ).run(imgs)

    shape = imgs[0].shape
    assert results[shape].paper_status == PaperDetectionStatus.NO_CANDIDATE
    assert results[shape].bbox == (0, 0, 200, 150)


def test_process_pebble_paper_group_by_shape_returns_no_candidate_when_paper_absent(make_group_metadata):
    """A shape group with no paper anywhere still gets a result, falling back to the full frame."""
    imgs = make_group_metadata(12, draw_paper=False)

    results = ProcessPebblePaperGroupByShape(
        SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
    ).run(imgs)

    shape = imgs[0].shape
    assert results[shape].paper_status == PaperDetectionStatus.NO_CANDIDATE
    assert results[shape].bbox == (0, 0, 200, 150)


def test_segment_cuttings_pebble_reuses_shared_group_crop_across_the_shape_group(make_group_metadata):
    """A large enough shape group is segmented via the shared group estimate, not per-image detection."""
    imgs = make_group_metadata(12)

    detections = segment_cuttings(
        imgs,
        config=SegmentationConfig(
            cuttings=SegmentationCuttingsConfig(
                downscale_factor=1.0,
                pebble_group=SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
            )
        ),
        cut_type="pebble",
    )

    assert len(detections) == len(imgs)
    for detection in detections:
        assert detection.cuttings is not None
        assert detection.cuttings.paper_status == PaperDetectionStatus.FOUND
        assert detection.cuttings.approach == ApproachType.GROUP


def test_segment_cuttings_pebble_falls_back_to_per_image_below_n_min_group(make_group_metadata):
    """A shape group too small for a shared estimate still gets segmented, per-image."""
    imgs = make_group_metadata(4)

    detections = segment_cuttings(
        imgs,
        config=SegmentationConfig(
            cuttings=SegmentationCuttingsConfig(
                downscale_factor=1.0,
                pebble_group=SegmentationCuttingsPebbleGroupConfig(downscale_factor=1.0, n_min_group=10),
            )
        ),
        cut_type="pebble",
    )

    assert len(detections) == len(imgs)
    for detection in detections:
        assert detection.cuttings is not None
        assert detection.cuttings.approach == ApproachType.SINGLE


def test_normalize_tray_scale_sets_median_size_without_touching_bbox():
    """Every result gets resize_to set to the batch's median width/height; bbox itself is untouched."""
    small = CuttingsSegmentResult(bbox=(0, 0, 100, 100))
    mid = CuttingsSegmentResult(bbox=(0, 0, 200, 200))
    big = CuttingsSegmentResult(bbox=(0, 0, 300, 300))

    _normalize_tray_scale([small, mid, big])

    assert small.bbox == (0, 0, 100, 100)
    assert mid.bbox == (0, 0, 200, 200)
    assert big.bbox == (0, 0, 300, 300)
    assert small.resize_to == mid.resize_to == big.resize_to == (200, 200)


def test_normalize_tray_scale_handles_empty_list():
    """An empty batch is a no-op, not a crash."""
    _normalize_tray_scale([])


def test_segment_cuttings_normalizes_tray_scale_across_batch(tmp_path):
    """Two differently-sized detected tray piles end up with the same resize_to (the batch median)."""
    small_pile = (150, 150, 250, 250)  # 100x100
    big_pile = (50, 50, 350, 350)  # 300x300
    small = _make_textured_metadata(tmp_path, 1.0, size=(400, 400), patches=[small_pile])
    big = _make_textured_metadata(tmp_path, 2.0, size=(400, 400), patches=[big_pile])

    detections = segment_cuttings(
        [small, big],
        config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
        cut_type="tray",
    )

    assert detections[0].cuttings is not None
    assert detections[1].cuttings is not None
    assert detections[0].cuttings.resize_to is not None
    assert detections[0].cuttings.resize_to == detections[1].cuttings.resize_to


def test_segment_cuttings_leaves_black_circle_crops_unnormalized(make_metadata):
    """black_circle has no fixed-size reference object, so crops keep their native detected size."""
    small = make_metadata(1.0, lambda draw: draw.ellipse((50, 50, 150, 150), fill=(200, 200, 200)))  # r=50
    big = make_metadata(2.0, lambda draw: draw.ellipse((20, 20, 280, 280), fill=(200, 200, 200)))  # r=130

    detections = segment_cuttings(
        [small, big],
        config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
        cut_type="black_circle",
    )

    def size(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
        return round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1])

    assert detections[0].cuttings is not None
    assert detections[1].cuttings is not None
    assert detections[0].cuttings.resize_to is None
    assert detections[1].cuttings.resize_to is None
    assert size(detections[0].cuttings.bbox) != size(detections[1].cuttings.bbox)


def test_segment_black_circle_falls_back_to_full_image_for_tiny_region(make_metadata):
    """A speck far below min_area_frac is treated as noise, not a real (if small) circle."""
    metadata = make_metadata(1.0, lambda draw: draw.ellipse((185, 185, 215, 215), fill=(200, 200, 200)))  # r=15

    result = segment_black_circle(metadata, SegmentationCuttingsConfig(downscale_factor=1.0))

    assert result.bbox == (0, 0, 400, 300)


def test_segment_tray_falls_back_to_full_image_for_tiny_region(tmp_path):
    """A textured region below min_area_frac is treated as noise, not a real pile."""
    tiny_patch = (50, 50, 90, 90)
    metadata = _make_textured_metadata(tmp_path, 1.0, size=(500, 400), patches=[tiny_patch])
    config = SegmentationCuttingsConfig(
        downscale_factor=1.0,
        tray=SegmentationCuttingsTrayConfig(min_area_frac=0.05),  # raised so the tiny patch counts as noise
    )

    result = segment_tray(metadata, config)

    assert result.bbox == (0, 0, 500, 400)


def test_is_full_frame_bbox():
    """The full-frame check tolerates float rounding but not a real crop."""
    shape = (300, 400, 3)  # (h, w, c)
    assert _is_full_frame_bbox((0, 0, 400, 300), shape)
    assert _is_full_frame_bbox((0.4, 0.0, 399.6, 300.0), shape)  # scale_bbox rounding slack
    assert not _is_full_frame_bbox((50, 50, 350, 250), shape)


def test_guard_degenerate_bbox_falls_back_to_full_image(make_metadata):
    """A sliver bbox (below min_crop_px) is replaced with the full image, preserving other fields."""
    metadata = make_metadata(1.0, size=(400, 300))
    sliver = CuttingsSegmentResult(bbox=(10, 10, 15, 200), time=1.0, paper_status=PaperDetectionStatus.FOUND)

    result = _guard_degenerate_bbox(metadata, sliver, min_crop_px=20)

    assert result.bbox == (0, 0, 400, 300)
    assert result.paper_status == PaperDetectionStatus.FOUND  # other fields are preserved


def test_guard_degenerate_bbox_leaves_real_crops_alone(make_metadata):
    """A crop at or above min_crop_px in both dimensions passes through unchanged."""
    metadata = make_metadata(1.0, size=(400, 300))
    real = CuttingsSegmentResult(bbox=(10, 10, 50, 50))

    result = _guard_degenerate_bbox(metadata, real, min_crop_px=20)

    assert result is real


def test_log_fallback_rate_reports_percentage(make_metadata, caplog):
    """The fallback-rate summary is logged unconditionally (not gated behind --mlflow)."""
    fell_back = make_metadata(1.0, size=(400, 300))
    real = make_metadata(2.0, size=(400, 300))
    segmented = [
        (fell_back, CuttingsSegmentResult(bbox=(0, 0, 400, 300))),
        (real, CuttingsSegmentResult(bbox=(10, 10, 100, 100))),
    ]

    with caplog.at_level("INFO"):
        _log_fallback_rate("black_circle", segmented)

    assert "1/2 images (50%) fell back" in caplog.text


def test_log_fallback_rate_skips_full_cut_type(make_metadata, caplog):
    """cut_type=full always has a 'full-frame' bbox by design, so the fallback rate is meaningless there."""
    metadata = make_metadata(1.0, size=(400, 300))
    segmented = [(metadata, CuttingsSegmentResult(bbox=(0, 0, 400, 300)))]

    with caplog.at_level("INFO"):
        _log_fallback_rate("full", segmented)

    assert caplog.text == ""


def test_log_fallback_rate_reports_pebble_status_counts(make_metadata, caplog):
    """For pebble specifically, the granular PaperDetectionStatus breakdown is also logged."""
    metadata = make_metadata(1.0, size=(400, 300))
    segmented = [
        (metadata, CuttingsSegmentResult(bbox=(0, 0, 400, 300), paper_status=PaperDetectionStatus.NO_CANDIDATE)),
    ]

    with caplog.at_level("INFO"):
        _log_fallback_rate("pebble", segmented)

    assert "no_candidate" in caplog.text


def test_log_crop_size_consistency_warns_on_high_variance(make_metadata, caplog):
    """Wildly different crop sizes across the batch trigger a warning, not just an info log."""
    segmented = [
        (make_metadata(float(i), size=(400, 300)), CuttingsSegmentResult(bbox=(0, 0, size, size)))
        for i, size in enumerate([20, 20, 20, 20, 200])  # one big outlier among small, consistent crops
    ]

    with caplog.at_level("INFO"):
        _log_crop_size_consistency("tray", segmented, cv_warn_threshold=0.3)

    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_log_crop_size_consistency_quiet_when_consistent(make_metadata, caplog):
    """Similarly-sized crops across the batch produce no warning."""
    segmented = [
        (make_metadata(float(i), size=(400, 300)), CuttingsSegmentResult(bbox=(0, 0, size, size)))
        for i, size in enumerate([100, 102, 98, 101, 99])
    ]

    with caplog.at_level("INFO"):
        _log_crop_size_consistency("tray", segmented, cv_warn_threshold=0.3)

    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_log_crop_size_consistency_skips_black_circle_and_tray_unrelated_types(make_metadata, caplog):
    """The consistency check only applies to black_circle/tray, which assume one fixed physical rig."""
    metadata = make_metadata(1.0, size=(400, 300))
    segmented = [(metadata, CuttingsSegmentResult(bbox=(0, 0, 10, 10)))]

    with caplog.at_level("INFO"):
        _log_crop_size_consistency("pebble", segmented, cv_warn_threshold=0.3)

    assert caplog.text == ""


def test_segment_cuttings_excludes_fallback_from_tray_normalization(tmp_path):
    """An uncropped fallback result doesn't skew (or get distorted by) the tray batch's shared resize_to."""
    real_a = _make_textured_metadata(tmp_path, 1.0, size=(400, 400), patches=[(150, 150, 250, 250)])  # 100x100
    real_b = _make_textured_metadata(tmp_path, 2.0, size=(400, 400), patches=[(140, 140, 260, 260)])  # 120x120
    blank = _make_textured_metadata(tmp_path, 3.0, size=(400, 400), patches=[])  # nothing to detect

    detections = segment_cuttings(
        [real_a, real_b, blank],
        config=SegmentationConfig(cuttings=SegmentationCuttingsConfig(downscale_factor=1.0)),
        cut_type="tray",
    )

    by_depth = {d.depth: d.cuttings for d in detections}
    cuttings_a, cuttings_b, cuttings_blank = by_depth[1.0], by_depth[2.0], by_depth[3.0]
    assert cuttings_a is not None
    assert cuttings_b is not None
    assert cuttings_blank is not None
    assert cuttings_blank.resize_to is None  # fallback result is left alone, not stretched to the tray's scale
    assert cuttings_a.resize_to == cuttings_b.resize_to  # real detections still share a common scale
