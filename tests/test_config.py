"""Tests for the config module."""

import pytest

from src.config import PipelineConfig, SegmentationConfig, StitchingConfig


def test_from_yaml_empty_file_uses_all_defaults(tmp_path):
    """An empty config file falls back to the dataclass defaults for both sections."""
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = PipelineConfig.from_yaml(path)

    assert config == PipelineConfig()


def test_from_yaml_overrides_only_specified_keys(tmp_path):
    """Keys present in the file override defaults; omitted keys keep their dataclass default."""
    path = tmp_path / "config.yaml"
    path.write_text("segmentation:\n  min_object_size: 42\n")

    config = PipelineConfig.from_yaml(path)

    assert config.segmentation.min_object_size == 42
    assert config.segmentation.opening_disk == SegmentationConfig().opening_disk
    assert config.stitching == StitchingConfig()


def test_from_yaml_reads_both_sections(tmp_path):
    """Both segmentation and stitching sections are read from the same file."""
    path = tmp_path / "config.yaml"
    path.write_text("segmentation:\n  tray_sat_threshold: 0.5\nstitching:\n  num_cores_per_image: 3\n")

    config = PipelineConfig.from_yaml(path)

    assert config.segmentation.tray_sat_threshold == 0.5
    assert config.stitching.num_cores_per_image == 3


def test_from_yaml_missing_section_uses_defaults(tmp_path):
    """A file that only specifies one section leaves the other section fully default."""
    path = tmp_path / "config.yaml"
    path.write_text("stitching:\n  output_width: 800\n")

    config = PipelineConfig.from_yaml(path)

    assert config.segmentation == SegmentationConfig()
    assert config.stitching.output_width == 800


def test_from_yaml_unknown_key_raises(tmp_path):
    """An unrecognized key in a section raises instead of being silently ignored."""
    path = tmp_path / "config.yaml"
    path.write_text("segmentation:\n  not_a_real_field: 1\n")

    with pytest.raises(TypeError):
        PipelineConfig.from_yaml(path)
