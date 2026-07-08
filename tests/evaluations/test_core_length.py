"""Tests for the core_length evaluation module."""

from pathlib import Path

import pytest

from evaluations.core_length import check_core_length
from src.config import CoreLengthCheckConfig
from src.models import CoreSegmentResult, ImageMetadataProcessed

# Depth intervals (in metres) with varied, exactly-representable values so the RANSAC
# fit has a well-defined slope regardless of which random subset it samples internally.
_INTERVALS_M = [0.5, 0.75, 1.0, 1.25, 1.5]
_RATIO_PX_PER_M = 800.0


def _make_detection(length: float, depth_start: float, interval: float) -> ImageMetadataProcessed:
    """Creates an ImageMetadataProcessed whose bounding box has the given length in pixels."""
    top = 0.0
    depth_end = depth_start + interval
    return ImageMetadataProcessed(
        borehole_id="GBC-CB50",
        depth_start=depth_start,
        depth_end=depth_end,
        image_path=Path(f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"),
        result=CoreSegmentResult(bounding_box=(100.0, top, 900.0, top + length)),
    )


def test_check_core_length_returns_empty_below_min_samples():
    """Too few detections to compute a reliable RANSAC fit means no results are returned."""
    detections = [_make_detection(800.0, depth_start=15.0 + i, interval=1.0) for i in range(3)]

    results = check_core_length(detections, CoreLengthCheckConfig(min_samples=5))

    assert results == []


def test_check_core_length_flags_only_the_outlier():
    """One core far outside the folder's fitted ratio is flagged; the rest pass.

    The outlier (2000px) is far enough from the clean line that RANSAC reliably excludes it
    from the fit, so the recovered ratio and the flagged outcome are deterministic.
    """
    lengths = [interval * _RATIO_PX_PER_M for interval in _INTERVALS_M[:4]] + [2000.0]
    detections = [
        _make_detection(length, depth_start=15.0 + i, interval=interval)
        for i, (length, interval) in enumerate(zip(lengths, _INTERVALS_M, strict=True))
    ]

    results = check_core_length(detections, CoreLengthCheckConfig(relative_tolerance=0.25, min_samples=5))

    assert [r.passed for r in results] == [True, True, True, True, False]
    assert results[-1].length_px == 2000.0
    assert results[-1].folder_ratio_px_per_m == pytest.approx(_RATIO_PX_PER_M)


def test_check_core_length_within_tolerance_all_pass():
    """Cores whose lengths only mildly deviate from the fitted ratio all pass."""
    lengths = [interval * _RATIO_PX_PER_M for interval in _INTERVALS_M[:4]] + [1450.0]
    detections = [
        _make_detection(length, depth_start=15.0 + i, interval=interval)
        for i, (length, interval) in enumerate(zip(lengths, _INTERVALS_M, strict=True))
    ]

    results = check_core_length(detections, CoreLengthCheckConfig(relative_tolerance=0.25, min_samples=5))

    assert all(r.passed for r in results)
    assert all(r.deviation < 0.25 for r in results)
