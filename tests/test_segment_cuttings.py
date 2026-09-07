"""Tests for the segment_cuttings module."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.config import SegmentationConfig
from src.models import ApproachType, ImageMetadataCuttings, PaperDetectionStatus
from src.segment.config import SegmentationCuttingsConfig, SegmentationCuttingsPebbleGroupConfig
from src.segment.segment_cuttings import segment_black_circle, segment_cuttings, segment_pebble
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
