"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.evaluations.config import CoreLengthCheckConfig, CoreWidthCheckConfig, EvaluationConfig


@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation step."""

    opening_disk: int = 20  # radius for binary_opening (removes noise)
    closing_disk: int = 20  # radius for binary_closing (fills gaps)
    foreground_blur_sigma: float = 5.0  # gaussian blur applied to each image for foreground detection.
    min_object_size: int = 500  # minimum blob size in pixels
    edge_margin_top: int = 100  # ignore top edge of image (ruler)
    edge_margin_bottom: int = 5  # ignore bottom edge of image (ruler)
    min_bbox_height: int = 500  # minimum height for a candidate core bounding box
    tray_sat_threshold: float = 0.28  # saturation above this = wooden tray (not rock)
    tray_sat_ratio: float = 0.75  # fraction of tray-saturated pixels in a row required to classify that row as tray
    min_size_for_bottom: int = 500_000  # minimum area for a candidate core to touch the bottom edge of the image
    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)


@dataclass
class StitchingConfig:
    """Tunable parameters for the stitching step."""

    num_cores_per_image: int = 6  # cores placed side by side per output sheet
    padding_vertical: int = 200  # top/bottom border height in pixels
    padding_horizontal: int = 150  # left/right border width in pixels
    ruler_width: int = 300  # width in pixels of each of the two depth rulers (left/right of the cores)
    core_height_px: int = 10000  # pixel budget for a core spanning core_height_m metres
    core_height_m: float = 1.0  # depth extent, in metres, that core_height_px pixels represents
    core_width_rerror: float = 1.5  # max allowed width ratio vs. the reference core before treated as an outlier
    font_size: int = 100  # font size (px) used for borehole ID, depth labels, and ruler tick labels


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
        raw_evaluation = raw.get("evaluation") or {}
        return cls(
            segmentation=SegmentationConfig(**(raw.get("segmentation") or {})),
            stitching=StitchingConfig(**(raw.get("stitching") or {})),
            evaluation=EvaluationConfig(
                core_width=CoreWidthCheckConfig(**(raw_evaluation.get("core_width") or {})),
                core_length=CoreLengthCheckConfig(**(raw_evaluation.get("core_length") or {})),
            ),
        )
