"""Tests for the config module."""

import textwrap

import pytest

from src.config import PipelineConfig, SegmentationConfig, SegmentationCoreConfig, StitchingConfig


def test_from_yaml_empty_file_uses_all_defaults(tmp_path):
    """An empty config file falls back to the dataclass defaults for both sections."""
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = PipelineConfig.from_yaml(path)

    assert config == PipelineConfig()


def test_from_yaml_reads_both_sections(tmp_path):
    """Keys present in the file override defaults; every other key keeps its dataclass default."""
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            segmentation:
              core:
                wood_sat_threshold: 0.5
            stitching:
              num_cores_per_image: 3
            """
        )
    )

    config = PipelineConfig.from_yaml(path)

    assert config.segmentation == SegmentationConfig(core=SegmentationCoreConfig(wood_sat_threshold=0.5))
    assert config.stitching == StitchingConfig(num_cores_per_image=3)


def test_from_yaml_unknown_key_raises(tmp_path):
    """An unrecognized key anywhere in the file raises instead of being silently ignored."""
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            segmentation:
              not_a_real_field: 1
            """
        )
    )

    with pytest.raises(TypeError):
        PipelineConfig.from_yaml(path)
