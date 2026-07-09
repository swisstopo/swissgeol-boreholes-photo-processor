"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
    padding_vertical: int = 95  # top/bottom border height in pixels
    padding_horizontal: int = 110  # left/right border width in pixels
    output_width: int = 1144  # output canvas width in pixels
    output_height: int = 1260  # output canvas height in pixels
    max_core_length_m: float = 1.0  # maximum core length in metres (fills the strip height exactly)


@dataclass
class CoreWidthCheckConfig:
    """Tunable parameters for the core width check evaluation."""

    relative_tolerance: float = 0.25  # flag if |width - folder_median| / folder_median exceeds this
    min_samples: int = 5  # below this, skip the check (median unreliable)


@dataclass
class CoreLengthCheckConfig:
    """Tunable parameters for the core length check evaluation."""

    relative_tolerance: float = 0.05  # ~5% buffer
    min_samples: int = 5  # below this, skip the RANSAC fit (too few points to be robust)


@dataclass
class EvaluationConfig:
    """Tunable parameters for the evaluation step."""

    core_width: CoreWidthCheckConfig = field(default_factory=CoreWidthCheckConfig)
    core_length: CoreLengthCheckConfig = field(default_factory=CoreLengthCheckConfig)


@dataclass
class CoreWidthCheckResults:
    """Results of the core width check evaluation."""

    filename: str  # name of the image file
    width: float  # width of the core in pixels
    folder_median_width: float  # median width of cores in the folder
    deviation: float  # relative deviation from the folder median width
    passed: bool  # whether the core passed the width check (True = within tolerance, False = flagged)


@dataclass
class CoreLengthCheckResults:
    """Results of the core length check evaluation."""

    filename: str  # name of the image file
    length_px: float  # length of the core in pixels
    expected_length_px: float  # (depth_end - depth_start) * folder_ratio_px_per_m
    folder_ratio_px_per_m: float  # RANSAC-fit slope for this folder
    deviation: float  # relative deviation from the expected length
    passed: bool  # whether the core passed the length check (True = within tolerance,


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
