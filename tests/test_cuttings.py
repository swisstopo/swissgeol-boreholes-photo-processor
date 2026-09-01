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
    """Only the first image (by filename) at a given depth is kept."""
    _write_image(tmp_path / "10m_00.jpg")
    _write_image(tmp_path / "10m_01.jpg")

    result = collect_cuttings(tmp_path)

    assert len(result) == 1
    assert result[0].image_path.name == "10m_00.jpg"


def test_collect_cuttings_dedup_keep_last(tmp_path):
    """With dedup_keep="last", the last image (by filename) at a given depth is kept."""
    _write_image(tmp_path / "10m_00.jpg")
    _write_image(tmp_path / "10m_01.jpg")

    result = collect_cuttings(tmp_path, dedup_keep="last")

    assert len(result) == 1
    assert result[0].image_path.name == "10m_01.jpg"


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
