from pathlib import Path

import pytest
from PIL import Image

from src.models import (
    CuttingsSegmentResult,
    ImageMetadataCores,
    ImageMetadataCuttings,
    ImageMetadataProcessedCuttings,
    PaperDetectionStatus,
)


@pytest.mark.parametrize(
    "path_str, borehole_id, depth_start, depth_end",
    [
        ("/data/GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF", "GBC-CB50", 15.0, 16.0),
        ("/data/Georessourcen/Pal-P1/Pal-P1_0000.00-0001.00_vd_p.TIF", "Pal-P1", 0.0, 1.0),
        ("/data/Handstuecke/Amsteg/Amsteg_108035.00-108036.00_vd_p.TIF", "Amsteg", 108035.0, 108036.0),
        ("/data/LBT/LBT-A1W/A1W_0000.00-0001.00_vd_p.TIF", "A1W", 0.0, 1.0),  # borehole_id from filename, not folder
        ("/data/LBT_Prognose/LBT-SB92-9B/LBT-SB92-9B_0069.60-0070.50_vd_p.TIF", "LBT-SB92-9B", 69.6, 70.5),
    ],
)
def test_from_path_parses_filename(path_str, borehole_id, depth_start, depth_end):
    path = Path(path_str)
    m = ImageMetadataCores.from_path(path)
    assert m.borehole_id == borehole_id
    assert m.depth_start == depth_start
    assert m.depth_end == depth_end
    assert m.folder == path.parent
    assert m.image_path == path
    assert isinstance(m.depth_start, float)
    assert isinstance(m.depth_end, float)


def test_from_path_raises_when_no_depth_in_filename():
    with pytest.raises(ValueError, match="No depth range found in filename"):
        ImageMetadataCores.from_path(Path("/data/GBC/GBC-CB50/some_random_file.TIF"))


def test_cuttings_segment_result_paper_status_counts_tallies_every_outcome():
    """Every PaperDetectionStatus value is reported, even at zero, and unset/failed results are skipped."""
    results = [
        CuttingsSegmentResult(bbox=(0, 0, 1, 1), paper_status=PaperDetectionStatus.FOUND),
        CuttingsSegmentResult(bbox=(0, 0, 1, 1), paper_status=PaperDetectionStatus.FOUND),
        CuttingsSegmentResult(bbox=(0, 0, 1, 1), paper_status=PaperDetectionStatus.NO_CANDIDATE),
        CuttingsSegmentResult(bbox=(0, 0, 1, 1)),  # black_circle method: paper_status is not applicable
        None,  # image that failed to segment
    ]

    counts = CuttingsSegmentResult.paper_status_counts(results)

    assert counts == {
        "found": 2,
        "no_candidate": 1,
        "degenerate_left_edge": 0,
        "cropped_too_much": 0,
    }


def test_cuttings_segment_result_to_dict_serializes_paper_status():
    """paper_status is serialized as its plain string value, or None when unset."""
    found = CuttingsSegmentResult(bbox=(0, 0, 1, 1), paper_status=PaperDetectionStatus.FOUND)
    unset = CuttingsSegmentResult(bbox=(0, 0, 1, 1))

    assert found.to_dict()["paper_status"] == "found"
    assert unset.to_dict()["paper_status"] is None


def test_load_cuttings_resizes_when_resize_to_is_set(tmp_path):
    """resize_to rescales the crop to that size, e.g. so trays of the same physical size stitch at one scale."""
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(100, 150, 200)).save(image_path)
    metadata = ImageMetadataCuttings(image_path=image_path, borehole_id="B", depth=1.0)
    cuttings = CuttingsSegmentResult(bbox=(0, 0, 100, 100), resize_to=(50, 25))

    crop = ImageMetadataProcessedCuttings.from_metadata(metadata, cuttings=cuttings).load_cuttings()

    assert crop.size == (50, 25)


def test_load_cuttings_keeps_native_crop_size_when_resize_to_is_none(tmp_path):
    """Without resize_to, the crop stays at whatever size the bbox itself defines."""
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(100, 150, 200)).save(image_path)
    metadata = ImageMetadataCuttings(image_path=image_path, borehole_id="B", depth=1.0)
    cuttings = CuttingsSegmentResult(bbox=(0, 0, 100, 60))

    crop = ImageMetadataProcessedCuttings.from_metadata(metadata, cuttings=cuttings).load_cuttings()

    assert crop.size == (100, 60)
