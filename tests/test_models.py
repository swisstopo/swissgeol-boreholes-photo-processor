from pathlib import Path

import pytest

from src.models import ImageMetadata


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
    m = ImageMetadata.from_path(path)
    assert m.borehole_id == borehole_id
    assert m.depth_start == depth_start
    assert m.depth_end == depth_end
    assert m.folder == path.parent
    assert m.image_path == path
    assert isinstance(m.depth_start, float)
    assert isinstance(m.depth_end, float)


def test_from_path_raises_when_no_depth_in_filename():
    with pytest.raises(ValueError, match="No depth range found in filename"):
        ImageMetadata.from_path(Path("/data/GBC/GBC-CB50/some_random_file.TIF"))
