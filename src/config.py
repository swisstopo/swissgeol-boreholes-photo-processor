"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation step."""

    opening_disk: int = 20  # radius for binary_opening (removes noise)
    closing_disk: int = 20  # radius for binary_closing (fills gaps)
    min_object_size: int = 500  # minimum blob size in pixels
    edge_margin_top: int = 100  # ignore top edge of image (ruler)
    edge_margin_bottom: int = 5  # ignore bottom edge of image (ruler)
    min_bbox_height: int = 500  # minimum height for a candidate core bounding box
    tray_sat_threshold: float = 0.28  # saturation above this = wooden tray (not rock)


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
class PipelineConfig:
    """Top-level configuration for the borehole photo processing pipeline."""

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file.

        Keys omitted from the file fall back to the defaults defined on
        SegmentationConfig / StitchingConfig.

        Args:
            path (Path): Path to the YAML config file.

        Returns:
            PipelineConfig: The loaded configuration.
        """
        raw = yaml.safe_load(path.read_text()) or {}
        return cls(
            segmentation=SegmentationConfig(**(raw.get("segmentation") or {})),
            stitching=StitchingConfig(**(raw.get("stitching") or {})),
        )
