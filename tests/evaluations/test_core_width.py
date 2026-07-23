"""Tests for the core_width evaluation module."""

from pathlib import Path

import pytest

from src.evaluations.config import CoreWidthCheckConfig
from src.evaluations.core import EvaluationWidthCompute
from src.models import CoreSegmentResult, ImageMetadataProcessed


def _make_detection(width: float, depth_start: float) -> ImageMetadataProcessed:
    """Creates an ImageMetadataProcessed whose bounding box has the given width in pixels.

    Width is the box's vertical (y) extent: raw TIF photos are landscape, with the depth
    axis running horizontally (x) and the core's physical width running vertically (y).
    """
    depth_end = depth_start + 1.0
    return ImageMetadataProcessed(
        borehole_id="GBC-CB50",
        depth_start=depth_start,
        depth_end=depth_end,
        image_path=Path(f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"),
        core=CoreSegmentResult(bbox=(100.0, 0.0, 1000.0, width)),
        tray=None,
        ruler=RulerSegmentResult(px_per_unit=100, bbox=(0.0, 0.0, 0.0, 0.0), bbox_units=[]),
    )


def test_check_core_width_returns_none_below_min_samples():
    """Too few detections to compute a reliable median means every core is skipped, not dropped."""
    detections = [_make_detection(800.0, depth_start=15.0 + i) for i in range(3)]

    results = EvaluationWidthCompute(config=CoreWidthCheckConfig(min_samples=5)).evaluate(detections)

    assert results == [None] * len(detections)


def test_check_core_width_flags_only_the_outlier():
    """One core far outside the folder's median width is flagged; the rest pass."""
    widths = [780.0, 790.0, 800.0, 810.0, 1500.0]  # last one is a clear outlier
    detections = [_make_detection(w, depth_start=15.0 + i) for i, w in enumerate(widths)]

    results = EvaluationWidthCompute(config=CoreWidthCheckConfig(relative_tolerance=0.25, min_samples=5)).evaluate(
        detections
    )

    assert [r.passed for r in results if r is not None] == [True, True, True, True, False]
    assert results[-1] is not None
    assert results[-1].measure == 15.0
    assert results[-1].reference == 8.0
    assert results[-1].relative_error == pytest.approx(0.875)  # (1500-800)/800


def test_check_core_width_deviation_exactly_at_tolerance_passes():
    """A deviation exactly equal to the tolerance is not flagged (comparison is a strict '>')."""
    widths = [100.0, 100.0, 100.0, 100.0, 125.0]  # folder median is 100.0; 125 deviates by exactly 25%
    detections = [_make_detection(w, depth_start=15.0 + i) for i, w in enumerate(widths)]

    results = EvaluationWidthCompute(config=CoreWidthCheckConfig(relative_tolerance=0.25, min_samples=5)).evaluate(
        detections
    )

    assert all(r is not None and r.passed for r in results)
