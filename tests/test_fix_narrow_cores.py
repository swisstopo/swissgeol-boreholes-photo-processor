"""Tests for the fix_narrow_cores module."""

from src.config import SegmentationConfig
from src.models import CoreSegmentResult, ImageMetadataProcessedCores


def test_fix_narrow_cores_overrides_when_conditions_met(make_metadata, draw_core, draw_wood):
    """Test that narrow cores are fixed when all conditions are met."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    # Create two images with identical dimensions
    # The scenario: image 2 has a core that's detected too high (narrow in height)
    # when it should actually be at the same position as image 1

    def draw_image_1(img):
        # Draw a normal core
        draw_core(img, (100, 200, 700, 600))
        # Draw wood on the sides
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        # Draw a core that's detected too high (narrow in height)
        # The actual core is at the same position as image 1, but detection made a mistake
        draw_core(img, (100, 250, 700, 550))  # Detected 50px higher and 50px lower
        # Draw wood on top and bottom (more wood visible because core is narrower in height)
        draw_wood(img, (0, 200, 800, 250))  # Wood on top
        draw_wood(img, (0, 550, 800, 600))  # Wood on bottom

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    # Create processed metadata with detected cores
    # Image 1: normal core bbox
    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    # Image 2: narrow core bbox (mistake) - detected too high
    # The detected bbox is (100, 250, 700, 550), but the actual core should be at (100, 200, 700, 600)
    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550)),
    )

    # Apply fix
    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was fixed
    assert len(fixed) == 2
    # The bbox should be overridden with the previous image's bbox
    # because when we apply (100, 200, 700, 600) to image 2, we get >= 25% wood
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 200, 700, 600)  # Should match previous image's bbox


def test_fix_narrow_cores_skips_when_dimensions_differ(make_metadata, draw_core, draw_wood):
    """Test that narrow cores are NOT fixed when image dimensions differ."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        draw_core(img, (100, 250, 700, 550))
        draw_wood(img, (0, 200, 800, 250))
        draw_wood(img, (0, 550, 800, 600))

    # Create images with DIFFERENT dimensions
    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1, size=(1200, 800))
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2, size=(1200, 900))  # Different height

    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550)),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was NOT fixed (dimensions differ)
    assert len(fixed) == 2
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 250, 700, 550)  # Should remain unchanged


def test_fix_narrow_cores_skips_when_difference_is_small(make_metadata, draw_core, draw_wood):
    """Test that narrow cores are NOT fixed when the difference is <= 10%."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        draw_core(img, (100, 210, 700, 610))  # Only 10px difference (< 10% of 400px height)
        draw_wood(img, (0, 210, 100, 610))
        draw_wood(img, (700, 210, 800, 610))

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 210, 700, 610)),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the core was NOT fixed (difference is too small)
    assert len(fixed) == 2
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 210, 700, 610)  # Should remain unchanged


def test_fix_narrow_cores_skips_when_wood_ratio_is_low(make_metadata, draw_core):
    """Test that narrow cores are NOT fixed when wood ratio is < 25%."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        # No wood on sides

    def draw_image_2(img):
        draw_core(img, (100, 250, 700, 550))
        # No wood on sides

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550)),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was NOT fixed (wood ratio is too low)
    assert len(fixed) == 2
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 250, 700, 550)  # Should remain unchanged


def test_fix_narrow_cores_uses_y_lines_when_available(make_metadata, draw_core, draw_wood):
    """Test that fix uses y_lines from current image when available and within 10%."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        draw_core(img, (100, 250, 700, 550))
        draw_wood(img, (0, 200, 800, 250))
        draw_wood(img, (0, 550, 800, 600))

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    # Image 1: normal core bbox
    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    # Image 2: narrow core bbox with y_lines that are close to the previous bbox
    # y_lines at 205 and 595 are within 10% of 200 and 600 (40px = 10% of 400px)
    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550), y_lines=[0, 205, 595, 800]),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was fixed using y_lines
    assert len(fixed) == 2
    # Should use y_lines 205 and 595 instead of 200 and 600
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 205, 700, 595)


def test_fix_narrow_cores_uses_prev_values_when_y_lines_too_far(make_metadata, draw_core, draw_wood):
    """Test that fix uses previous bbox values when y_lines are too far away."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        draw_core(img, (100, 250, 700, 550))
        draw_wood(img, (0, 200, 800, 250))
        draw_wood(img, (0, 550, 800, 600))

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    # Image 1: normal core bbox
    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    # Image 2: narrow core bbox with y_lines that are too far from the previous bbox
    # y_lines at 300 and 500 are more than 10% away from 200 and 600
    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550), y_lines=[0, 300, 500, 800]),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was fixed using previous values (not y_lines)
    assert len(fixed) == 2
    # Should use previous bbox values 200 and 600 because y_lines are too far
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 200, 700, 600)


def test_fix_narrow_cores_handles_missing_y_lines(make_metadata, draw_core, draw_wood):
    """Test that fix handles missing y_lines gracefully."""
    from src.segment.fix_narrow_cores import fix_narrow_cores

    def draw_image_1(img):
        draw_core(img, (100, 200, 700, 600))
        draw_wood(img, (0, 200, 100, 600))
        draw_wood(img, (700, 200, 800, 600))

    def draw_image_2(img):
        draw_core(img, (100, 250, 700, 550))
        draw_wood(img, (0, 200, 800, 250))
        draw_wood(img, (0, 550, 800, 600))

    metadata_1 = make_metadata(15.0, 16.0, draw_fn=draw_image_1)
    metadata_2 = make_metadata(16.0, 17.0, draw_fn=draw_image_2)

    # Image 1: normal core bbox
    detection_1 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_1,
        core=CoreSegmentResult(bbox=(100, 200, 700, 600)),
    )

    # Image 2: narrow core bbox with no y_lines (None)
    detection_2 = ImageMetadataProcessedCores.from_metadata(
        metadata=metadata_2,
        core=CoreSegmentResult(bbox=(100, 250, 700, 550), y_lines=None),
    )

    config = SegmentationConfig()
    fixed = fix_narrow_cores([detection_1, detection_2], config)

    # Check that the narrow core was fixed using previous values (no y_lines available)
    assert len(fixed) == 2
    assert fixed[1].core is not None
    assert fixed[1].core.bbox == (100, 200, 700, 600)
