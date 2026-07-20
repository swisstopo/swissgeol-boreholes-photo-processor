"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class SegmentationError(Exception):
    """Raised when segmentation fails for a single image."""


@dataclass
class SegmentationCoreConfig:
    """Tunable parameters for trimming the wooden tray off a detected core bounding box."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    tray_sat_ratio: float = 0.75  # fraction of tray-saturated pixels in a row required to classify that row as tray
    tray_sat_threshold: float = 0.28  # saturation above this = wooden tray (not rock)


@dataclass
class SegmentationRulerConfig:
    """Tunable parameters for detecting a depth ruler via OCR on its printed number ticks."""

    downscale_factor: float = 0.5  # Scale images by this factor before OCR
    text_min_value: int = 1  # Minimum visible number on ruler
    text_max_value: int = 99  # Maximum visible number on ruler
    r_error_outliers: float = 0.1  # Allow 10% error for inliers detection


@dataclass
class SegmentationTrayMultipleConfig:
    """Tunable parameters for estimating a shared tray/core bounding box across a batch of images."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    foreground_blur_sigma: float = 5.0  # gaussian blur applied to each image for foreground detection.
    n_min_foreground: int = 10  # min images required to estimate a foreground; also the sample size drawn per group
    seed: int = 0  # seed for randomly sampling images from a group


@dataclass
class SegmentationTraySingleConfig:
    """Tunable parameters for segmenting a single image via thresholding when no shared bbox is available."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    closing_disk: int = 20  # radius for binary_closing (fills gaps)
    edge_margin_bottom: int = 5  # ignore bottom edge of image (ruler)
    edge_margin_top: int = 100  # ignore top edge of image (ruler)
    min_bbox_height: int = 500  # minimum height for a candidate core bounding box
    min_object_size: int = 500  # minimum blob size in pixels
    min_size_for_bottom: int = 500_000  # minimum area for a candidate core to touch the bottom edge of the image
    opening_disk: int = 20  # radius for binary_opening (removes noise)


@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation step."""

    core: SegmentationCoreConfig = field(default_factory=SegmentationCoreConfig)
    ruler: SegmentationRulerConfig = field(default_factory=SegmentationRulerConfig)
    tray_multiple: SegmentationTrayMultipleConfig = field(default_factory=SegmentationTrayMultipleConfig)
    tray_single: SegmentationTraySingleConfig = field(default_factory=SegmentationTraySingleConfig)


@dataclass
class StitchingConfig:
    """Tunable parameters for the stitching step."""

    core_height_m: float = 1.0  # depth extent, in metres, that core_height_px pixels represents
    core_height_px: int = 10000  # pixel budget for a core spanning core_height_m metres
    core_width_rerror: float = 1.5  # max allowed width ratio vs. the reference core before treated as an outlier
    font_size: int = 100  # font size (px) used for borehole ID, depth labels, and ruler tick labels
    num_cores_per_image: int = 6  # cores placed side by side per output sheet
    padding_horizontal: int = 150  # left/right border width in pixels
    padding_vertical: int = 200  # top/bottom border height in pixels
    ruler_width: int = 300  # width in pixels of each of the two depth rulers (left/right of the cores)


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
        segmentation_raw = dict(raw.pop("segmentation", None) or {})
        return cls(
            segmentation=SegmentationConfig(
                core=SegmentationCoreConfig(**(segmentation_raw.pop("core", None) or {})),
                ruler=SegmentationRulerConfig(**(segmentation_raw.pop("ruler", None) or {})),
                tray_multiple=SegmentationTrayMultipleConfig(**(segmentation_raw.pop("tray_multiple", None) or {})),
                tray_single=SegmentationTraySingleConfig(**(segmentation_raw.pop("tray_single", None) or {})),
                **segmentation_raw,
            ),
            stitching=StitchingConfig(**(raw.pop("stitching", None) or {})),
            **raw,
        )
