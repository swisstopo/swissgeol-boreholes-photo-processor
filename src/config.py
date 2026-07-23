"""Configuration models and loading for the borehole photo processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.evaluations.config import CoreLengthCheckConfig, CoreWidthCheckConfig, EvaluationConfig


@dataclass
class SegmentationCoreConfig:
    """Tunable parameters for trimming the wooden tray off."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    background_val_hratio: float = 0.75  # fraction of dark-background pixels in a row required to trim that row
    background_val_vratio: float = 0.90  # fraction of dark-background pixels in a column required to trim that column
    background_val_threshold: float = 0.20  # value (HSV) below this = black background (not rock)
    wood_sat_hratio: float = 0.75  # fraction of tray-saturated pixels in a row required to classify that row as tray
    wood_sat_threshold: float = 0.28  # saturation above this = wooden tray (not rock)
    min_segment_height_px: int = 10  # minimum height (px) for a left/right segment to be kept as core


@dataclass
class SegmentationRulerConfig:
    """Tunable parameters for detecting a depth ruler via OCR on its printed number ticks."""

    downscale_factor: float = 0.5  # scale images by this factor before OCR
    text_min_value: int = 1  # minimal visible number on ruler
    text_max_value: int = 99  # maximal visible number on ruler
    r_error_outliers: float = 0.1  # allow 10% error for inliers detection


@dataclass
class SegmentationTrayMultipleConfig:
    """Tunable parameters for estimating a shared tray/core bounding box across a batch of images."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    foreground_blur_sigma: float = 5.0  # gaussian blur applied to each image for foreground detection.
    n_min_foreground: int = 10  # minimum number images required to estimate a foreground


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

    font_size: int = 100  # font size (px) used for borehole ID, depth labels, and ruler tick labels
    max_core_height: int = 10000  # cap on each core crop's resized height (pixels) and the canvas row height
    max_core_width: int = 1200  # cap on each core crop's resized width (pixels) and the per-core column width
    num_cores_per_image: int = 6  # cores placed side by side per output sheet
    padding_horizontal: int = 150  # left/right border width in pixels
    padding_vertical: int = 200  # top/bottom border height in pixels
    ruler_width: int = 300  # width in pixels of each of the two depth rulers (left/right of the cores)


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
        raw_evaluation = dict(raw.pop("evaluation", None) or {})
        return cls(
            segmentation=SegmentationConfig(
                core=SegmentationCoreConfig(**(raw_segmentation.pop("core", None) or {})),
                ruler=SegmentationRulerConfig(**(raw_segmentation.pop("ruler", None) or {})),
                tray_multiple=SegmentationTrayMultipleConfig(**(raw_segmentation.pop("tray_multiple", None) or {})),
                tray_single=SegmentationTraySingleConfig(**(raw_segmentation.pop("tray_single", None) or {})),
                **raw_segmentation,
            ),
            stitching=StitchingConfig(**(raw.pop("stitching", None) or {})),
            evaluation=EvaluationConfig(
                core_width=CoreWidthCheckConfig(**(raw_evaluation.pop("core_width", None) or {})),
                core_length=CoreLengthCheckConfig(**(raw_evaluation.pop("core_length", None) or {})),
                **raw_evaluation,
            ),
            **raw,
        )
