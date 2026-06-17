from pathlib import Path

import pytest

from src.models import ImageMetadata


class TestImageMetadataFromPath:
    """Tests for ImageMetadata.from_path() filename parsing."""

    def test_parses_depths_from_standard_filename(self):
        path = Path("/data/GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.depth_start == 15.0
        assert m.depth_end == 16.0

    def test_derives_borehole_id_from_filename(self):
        path = Path("/data/GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.borehole_id == "GBC-CB50"
        assert m.folder == Path("/data/GBC/GBC-CB50")
        assert m.image_path == path

    def test_parses_borehole_id_with_hyphen(self):
        path = Path("/data/Georessourcen/Pal-P1/Pal-P1_0000.00-0001.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.borehole_id == "Pal-P1"
        assert m.depth_start == 0.0
        assert m.depth_end == 1.0

    def test_parses_large_depth_values(self):
        path = Path("/data/Handstuecke/Amsteg/Amsteg_108035.00-108036.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.depth_start == 108035.0
        assert m.depth_end == 108036.0

    def test_borehole_id_comes_from_filename_not_folder(self):
        # folder is LBT-A1W but filename prefix is A1W — borehole_id comes from the filename
        path = Path("/data/LBT/LBT-A1W/A1W_0000.00-0001.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.borehole_id == "A1W"
        assert m.depth_start == 0.0
        assert m.depth_end == 1.0

    def test_parses_borehole_id_with_hyphen_and_numbers(self):
        path = Path("/data/LBT_Prognose/LBT-SB92-9B/LBT-SB92-9B_0069.60-0070.50_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert m.borehole_id == "LBT-SB92-9B"
        assert m.depth_start == 69.6
        assert m.depth_end == 70.5

    def test_depth_fields_are_floats(self):
        path = Path("/data/GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")
        m = ImageMetadata.from_path(path)
        assert isinstance(m.depth_start, float)
        assert isinstance(m.depth_end, float)

    def test_raises_when_no_depth_in_filename(self):
        path = Path("/data/GBC/GBC-CB50/some_random_file.TIF")
        with pytest.raises(ValueError, match="No depth range found in filename"):
            ImageMetadata.from_path(path)
