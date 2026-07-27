"""Tests for the check_core per-file result merging."""

from pathlib import Path

from src.evaluations.config import CoreLengthCheckConfig, CoreWidthCheckConfig, EvaluationConfig
from src.evaluations.core import evaluate_detections
from src.models import CoreSegmentResult, ImageMetadataProcessed, RulerSegmentResult


def _make_detection(width: float, length: float, depth_start: float, interval: float = 1.0) -> ImageMetadataProcessed:
    """Creates an ImageMetadataProcessed with the given bounding box width/length in pixels."""
    left = 100.0
    depth_end = depth_start + interval
    return ImageMetadataProcessed(
        borehole_id="GBC-CB50",
        depth_start=depth_start,
        depth_end=depth_end,
        image_path=Path(f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"),
        core=CoreSegmentResult(bbox=(left, 0.0, left + length, width)),
        tray=None,
        ruler=RulerSegmentResult(px_per_unit=100, bbox=(0.0, 0.0, 0.0, 0.0), bbox_units=[]),
    )


def test_check_core_merges_width_and_length_per_file():
    """Each file's result nests both its width and length check results, keyed by filename."""
    detections = [_make_detection(width=800.0, length=800.0, depth_start=15.0 + i) for i in range(5)]

    results = evaluate_detections(
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
        assert result.length is not None


def test_check_core_keeps_filename_alignment_when_one_core_is_skipped():
    """A single zero-length interval must not shift width/length results onto the wrong file.

    check_core_length skips only the individual core whose depth interval is zero (its expected
    length is undefined), while every other file in the folder is still checked -- this must not
    misattribute results to the wrong file or drop it from the output.
    """
    detections = [_make_detection(width=800.0, length=800.0, depth_start=15.0 + i) for i in range(6)]
    detections[2] = _make_detection(width=800.0, length=800.0, depth_start=17.0, interval=0.0)

    results = evaluate_detections(
        detections,
        EvaluationConfig(
            core_width=CoreWidthCheckConfig(min_samples=5),
            core_length=CoreLengthCheckConfig(min_samples=5),
        ),
    )

    assert len(results) == 6
    for detection, result in zip(detections, results, strict=True):
        assert result.filename == detection.image_path.name
    assert results[2].length is None
    assert all(result.length is not None for result in results[:2] + results[3:])


def test_check_core_leaves_width_or_length_none_below_min_samples():
    """A check below its own min_samples contributes None instead of dropping the file."""
    detections = [_make_detection(width=800.0, length=800.0, depth_start=15.0 + i) for i in range(5)]

    results = evaluate_detections(
        detections,
        EvaluationConfig(
            core_width=CoreWidthCheckConfig(min_samples=100),  # too few samples: width check skipped
            core_length=CoreLengthCheckConfig(min_samples=5),
        ),
    )

    assert len(results) == 5
    assert all(r.width is None for r in results)
    assert all(r.length is not None for r in results)
