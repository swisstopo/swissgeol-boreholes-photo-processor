"""Tests for the preprocessing.cuttings module."""

from pathlib import Path

from PIL import Image

from src.preprocessing.cuttings import collect_cuttings


def _write_image(path: Path, size: tuple[int, int] = (10, 10)) -> None:
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_collect_cuttings_sorts_by_depth(tmp_path):
    """Images are returned sorted by the depth parsed from their filename, not folder order."""
    _write_image(tmp_path / "20m_00.jpg")
    _write_image(tmp_path / "5m_00.jpg")
    _write_image(tmp_path / "10m_00.jpg")

    result = collect_cuttings(tmp_path)

    assert [m.depth for m in result] == [5.0, 10.0, 20.0]
    assert all(m.borehole_id == tmp_path.name for m in result)


def test_collect_cuttings_drops_duplicate_depth(tmp_path):
    """dedup_keep controls whether the first or last image (by filename) at a given depth is kept."""
    _write_image(tmp_path / "10m_00.jpg")
    _write_image(tmp_path / "10m_01.jpg")

    result_first = collect_cuttings(tmp_path, dedup_keep="first")
    result_last = collect_cuttings(tmp_path, dedup_keep="last")

    assert result_first[0].image_path.name == "10m_00.jpg"
    assert result_last[0].image_path.name == "10m_01.jpg"


def test_collect_cuttings_prefers_narrower_span_for_montagny_range_duplicates(tmp_path):
    """A wide-span pre-existing composite is dropped in favor of the real narrow-span photo."""
    _write_image(tmp_path / "MONTAGNY-2_Cuttings_1335.00-1400.00.jpg")  # wide composite, 65m span
    _write_image(tmp_path / "MONTAGNY-2_Cuttings_1395.00-1400.00.jpg")  # real photo, 5m span

    result = collect_cuttings(tmp_path)

    assert len(result) == 1
    assert result[0].image_path.name == "MONTAGNY-2_Cuttings_1395.00-1400.00.jpg"


def test_collect_cuttings_prefers_narrower_span_regardless_of_filename_order(tmp_path):
    """The narrow-span image wins even when it sorts before the wide one alphabetically."""
    _write_image(tmp_path / "MONTAGNY-2_Cuttings_0395.00-0400.00.jpg")  # real photo, 5m span
    _write_image(tmp_path / "MONTAGNY-2_Cuttings_0300.00-0400.00.jpg")  # wide composite, 100m span

    result = collect_cuttings(tmp_path)

    assert len(result) == 1
    assert result[0].image_path.name == "MONTAGNY-2_Cuttings_0395.00-0400.00.jpg"


def test_collect_cuttings_override_wins_over_duplicate_depth(tmp_path):
    """An override_ file always wins at its depth, regardless of dedup_keep or filename order."""
    _write_image(tmp_path / "10m_00.jpg")
    _write_image(tmp_path / "override_10m_01.jpg")

    result = collect_cuttings(tmp_path, dedup_keep="first")

    assert len(result) == 1
    assert result[0].image_path.name == "override_10m_01.jpg"


def test_collect_cuttings_skips_vial_photos(tmp_path):
    """Sample-vial photos (no real depth) are excluded outright, not parsed as depth 0."""
    _write_image(tmp_path / "00-Vials-IMG_20240525_084316.jpg")
    _write_image(tmp_path / "10m_00.jpg")

    result = collect_cuttings(tmp_path)

    assert [m.depth for m in result] == [10.0]


def test_collect_cuttings_skips_vue_generale_photos(tmp_path):
    """Vue-generale overview photos (no real depth) are excluded outright, case-insensitively."""
    _write_image(tmp_path / "10m_Vue-Generale.jpg")
    _write_image(tmp_path / "20m_00.jpg")

    result = collect_cuttings(tmp_path)

    assert [m.depth for m in result] == [20.0]


def test_collect_cuttings_skips_unreadable_file_without_crashing(tmp_path):
    """A file with a cuttings extension but no real image content is skipped, not fatal."""
    (tmp_path / "10m_00.jpg").write_text("not an image")
    _write_image(tmp_path / "20m_00.jpg")

    result = collect_cuttings(tmp_path)

    assert [m.depth for m in result] == [20.0]
