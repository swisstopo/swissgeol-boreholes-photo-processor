"""Tests for the check_core per-file result merging."""

from pathlib import Path

from evaluations.core import check_core
from src.config import CoreLengthCheckConfig, CoreWidthCheckConfig, EvaluationConfig
from src.models import CoreSegmentResult, ImageMetadataProcessed


def _make_detection(width: float, length: float, depth_start: float, interval: float = 1.0) -> ImageMetadataProcessed:
    """Creates an ImageMetadataProcessed with the given bounding box width/length in pixels."""
    left = 100.0
    depth_end = depth_start + interval
    return ImageMetadataProcessed(
        borehole_id="GBC-CB50",
        depth_start=depth_start,
        depth_end=depth_end,
        image_path=Path(f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"),
        result=CoreSegmentResult(bounding_box=(left, 0.0, left + length, width)),
    )


def test_check_core_merges_width_and_length_per_file():
    """Each file's result nests both its width and length check results, keyed by filename."""
    detections = [_make_detection(width=800.0, length=800.0, depth_start=15.0 + i) for i in range(5)]

    results = check_core(
        detections,
        EvaluationConfig(
            core_width=CoreWidthCheckConfig(min_samples=5),
            core_length=CoreLengthCheckConfig(min_samples=5),
        ),
    )

    assert len(results) == 5
    for detection, result in zip(detections, results, strict=True):
        assert result.filename == detection.image_path.name
        assert result.width is not None
        assert result.width.filename == result.filename
        assert result.length is not None
        assert result.length.filename == result.filename


def test_check_core_leaves_width_or_length_none_below_min_samples():
    """A check below its own min_samples contributes None instead of dropping the file."""
    detections = [_make_detection(width=800.0, length=800.0, depth_start=15.0 + i) for i in range(5)]

    results = check_core(
        detections,
        EvaluationConfig(
            core_width=CoreWidthCheckConfig(min_samples=100),  # too few samples: width check skipped
            core_length=CoreLengthCheckConfig(min_samples=5),
        ),
    )

    assert len(results) == 5
    assert all(r.width is None for r in results)
    assert all(r.length is not None for r in results)
