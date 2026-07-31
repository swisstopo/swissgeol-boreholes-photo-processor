"""Tests for the core_length evaluation module."""

from pathlib import Path

from src.evaluations.config import CoreLengthCheckConfig
from src.evaluations.core import EvaluationLengthCompute
from src.models import CoreSegmentResult, ImageMetadataProcessed, RulerSegmentResult

# Depth intervals (in metres) with varied, exactly-representable values.
_INTERVALS_M = [0.5, 0.75, 1.0, 1.25, 1.5]
_RATIO_PX_PER_M = 800.0


def _make_detection(length: float, depth_start: float, interval: float) -> ImageMetadataProcessed:
    """Creates an ImageMetadataProcessed whose bounding box has the given length in pixels.

    Length is the box's horizontal (x) extent: raw TIF photos are landscape, with the depth
    axis running horizontally (x) and the core's physical width running vertically (y).
    """
    left = 100.0
    depth_end = depth_start + interval
    return ImageMetadataProcessed(
        borehole_id="GBC-CB50",
        depth_start=depth_start,
        depth_end=depth_end,
        image_path=Path(f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"),
        core=CoreSegmentResult(bbox=(left, 0.0, left + length, 900.0)),
        tray=None,
        ruler=RulerSegmentResult(px_per_unit=100, bbox=(0.0, 0.0, 0.0, 0.0), bbox_units=[]),
    )


def test_check_core_length_returns_none_below_min_samples():
    """Too few detections to compute a reliable median ratio means every core is skipped, not dropped."""
    detections = [_make_detection(800.0, depth_start=15.0 + i, interval=1.0) for i in range(3)]

    results = EvaluationLengthCompute(CoreLengthCheckConfig(min_samples=5)).evaluate(detections)

    assert results == [None] * len(detections)


def test_check_core_length_flags_only_the_outlier():
    """One core far outside the folder's fitted ratio is flagged; the rest pass.

    The outlier (2000px) is a minority (1 of 5) of the per-core ratios, so the median
    reliably excludes it, keeping the recovered ratio and flagged outcome deterministic.
    """
    lengths = [interval * _RATIO_PX_PER_M for interval in _INTERVALS_M[:4]] + [2000.0]
    detections = [
        _make_detection(length, depth_start=15.0 + i, interval=interval)
        for i, (length, interval) in enumerate(zip(lengths, _INTERVALS_M, strict=True))
    ]

    results = EvaluationLengthCompute(CoreLengthCheckConfig(relative_tolerance=0.25, min_samples=5)).evaluate(
        detections
    )

    assert [r.passed for r in results if r is not None] == [True, True, True, True, False]
    assert results[-1] is not None
    assert results[-1].measure == 20.0
