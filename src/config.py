"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.evaluations.config import CoreLengthCheckConfig, CoreWidthCheckConfig, EvaluationConfig
from src.segment.config import (
    SegmentationConfig,
    SegmentationCoreConfig,
    SegmentationRulerConfig,
    SegmentationTrayGroupConfig,
    SegmentationTraySingleConfig,
)
from src.stitching.config import CoreStitchingConfig, CuttingsStitchingConfig, StitchingConfig


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


@dataclass
class PipelineConfig:
    """Top-level configuration for the borehole photo processing pipeline."""

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file.

        Keys omitted from the file fall back to the defaults defined on
        SegmentationConfig / StitchingConfig / EvaluationConfig.

        Args:
            path (Path): Path to the YAML config file.

        Returns:
            PipelineConfig: The loaded configuration.
        """
        raw = yaml.safe_load(path.read_text()) or {}
        raw_segmentation = dict(raw.pop("segmentation", None) or {})
        raw_stitching = dict(raw.pop("stitching", None) or {})
        raw_evaluation = dict(raw.pop("evaluation", None) or {})
        return cls(
            segmentation=SegmentationConfig(
                core=SegmentationCoreConfig(**(raw_segmentation.pop("core", None) or {})),
                ruler=SegmentationRulerConfig(**(raw_segmentation.pop("ruler", None) or {})),
                tray_group=SegmentationTrayGroupConfig(**(raw_segmentation.pop("tray_group", None) or {})),
                tray_single=SegmentationTraySingleConfig(**(raw_segmentation.pop("tray_single", None) or {})),
                **raw_segmentation,
            ),
            stitching=StitchingConfig(
                core=CoreStitchingConfig(**(raw_stitching.pop("core", None) or {})),
                cuttings=CuttingsStitchingConfig(**(raw_stitching.pop("cuttings", None) or {})),
                **raw_stitching,
            ),
            evaluation=EvaluationConfig(
                core_width=CoreWidthCheckConfig(**(raw_evaluation.pop("core_width", None) or {})),
                core_length=CoreLengthCheckConfig(**(raw_evaluation.pop("core_length", None) or {})),
                **raw_evaluation,
            ),
            **raw,
        )
