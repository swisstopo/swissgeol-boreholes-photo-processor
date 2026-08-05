"""Tests for the config module."""

import textwrap

import pytest

from src.config import (
    CoreStitchingConfig,
    PipelineConfig,
    SegmentationCoreConfig,
    SegmentationRulerConfig,
    SegmentationTrayGroupConfig,
    SegmentationTraySingleConfig,
    StitchingConfig,
)
from src.evaluations.config import CoreLengthCheckConfig, CoreWidthCheckConfig


def test_from_yaml_empty_file_uses_all_defaults(tmp_path):
    """An empty config file falls back to the dataclass defaults for both sections."""
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = PipelineConfig.from_yaml(path)

    assert config == PipelineConfig()


def test_from_yaml_reads_all_sections(tmp_path):
    """Every nested sub-config (segmentation.*, stitching, evaluation.*) applies its own overrides independently."""
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            segmentation:
              core:
                wood_sat_threshold: 0.5
              ruler:
                text_max_value: 50
              tray_group:
                n_min_foreground: 50
              tray_single:
                min_bbox_height: 200
            stitching:
              core:
                max_core_width: 900
                max_core_height: 5000
            evaluation:
              core_width:
                min_samples: 100
              core_length:
                min_samples: 101
            """
        )
    )

    config = PipelineConfig.from_yaml(path)

    assert config.segmentation.core == SegmentationCoreConfig(wood_sat_threshold=0.5)
    assert config.segmentation.ruler == SegmentationRulerConfig(text_max_value=50)
    assert config.segmentation.tray_group == SegmentationTrayGroupConfig(n_min_foreground=50)
    assert config.segmentation.tray_single == SegmentationTraySingleConfig(min_bbox_height=200)
    assert config.stitching == StitchingConfig(core=CoreStitchingConfig(max_core_width=900, max_core_height=5000))
    assert config.evaluation.core_width == CoreWidthCheckConfig(min_samples=100)
    assert config.evaluation.core_length == CoreLengthCheckConfig(min_samples=101)


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
