"""Configuration and result models for segmentation pipeline."""

from dataclasses import dataclass, field


@dataclass
class SegmentationCoreTrimConfig:
    """Tunable parameters for trimming the wooden tray and black background off, and filtering thin segments."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    background_val_hratio: float = 0.90  # fraction of dark-background pixels in a row required to trim that row
    background_val_vratio: float = 0.70  # fraction of dark-background pixels in a column required to trim that column
    background_val_threshold: float = 0.15  # value (HSV) below this = black background (not rock)
    wood_sat_hratio: float = 0.75  # fraction of tray-saturated pixels in a row required to classify that row as tray
    wood_sat_threshold: float = 0.28  # saturation above this = wooden tray (not rock)
    min_segment_height_px: int = 100  # minimum height (px) for a left/right segment to be kept as core


@dataclass
class SegmentationRulerConfig:
    """Tunable parameters for detecting a depth ruler via OCR on its printed number ticks."""

    downscale_factor: float = 0.5  # scale images by this factor before OCR
    text_min_value: int = 1  # minimal visible number on ruler
    text_max_value: int = 99  # maximal visible number on ruler
    r_error_outliers: float = 0.1  # allow 10% error for outliers detection
    n_min_ruler: int = 10  # images to OCR per shape group before taking the median-scale detection
    seed: int = 0  # seed for randomly sampling images from a group


@dataclass
class SegmentationTrayGroupConfig:
    """Tunable parameters for estimating a shared tray/core bounding box across a batch of images."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    foreground_blur_sigma: float = 5.0  # gaussian blur applied to each image for foreground detection
    max_flow_shift: int = 100  # max optical flow displacement to allow, in pixels
    n_min_foreground: int = 10  # minimum number images required to estimate a foreground
    seed: int = 0  # seed for randomly sampling images from a group


@dataclass
class SegmentationTraySingleConfig:
    """Tunable parameters for segmenting a single image via thresholding when no shared bbox is available."""

    downscale_factor: float = 0.125  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    block_size: int = 4000  # pixel neighborhood used to calculate the threshold
    closing_disk: int = 50  # radius for binary_closing (fills gaps)
    edge_margin_bottom: int = 5  # ignore bottom edge of image (ruler)
    edge_margin_top: int = 100  # ignore top edge of image (ruler)
    min_bbox_height: int = 500  # minimum height for a candidate core bounding box
    min_object_size: int = 100_000  # minimum blob size in pixels
    min_size_for_bottom: int = 500_000  # minimum area for a candidate core to touch the bottom edge of the image
    opening_disk: int = 20  # radius for binary_opening (removes noise)


@dataclass
class SegmentationCoreConfig:
    """Tunable parameters for the core segmentation step."""

    core: SegmentationCoreTrimConfig = field(default_factory=SegmentationCoreTrimConfig)
    ruler: SegmentationRulerConfig = field(default_factory=SegmentationRulerConfig)
    tray_group: SegmentationTrayGroupConfig = field(default_factory=SegmentationTrayGroupConfig)
    tray_single: SegmentationTraySingleConfig = field(default_factory=SegmentationTraySingleConfig)


@dataclass
class SegmentationCuttingsPebbleConfig:
    """Tunable parameters for segmenting pebble cuttings via paper-sheet detection."""

    min_extent: float = 0.45  # minimum bounding-box fill fraction for a paper candidate
    min_solidity: float = 0.7  # minimum convexity for a paper candidate
    min_area_frac: float = 0.002  # minimum paper candidate area, as a fraction of the image area
    max_area_frac: float = 0.35  # maximum paper candidate area, as a fraction of the image area
    edge_margin: float = 0.03  # fraction of the relevant dimension a candidate must reach to count as edge-anchored
    closing_disk: int = 25  # radius for binary closing that fills the black stripes punched into the paper mask
    sat_threshold: float = 0.15  # saturation (HSV) below this = colorless (candidate paper)
    val_threshold_strict: float = 0.85  # brightness (HSV) above this = bright paper, tried first
    val_threshold_loose: float = 0.45  # looser brightness fallback for darker exposures, tried if the strict one fails
    stripe_min_drop: float = 0.2  # brightness drop (vs. the patch's 85th percentile) that counts as a dark stripe
    stripe_max_width_frac: float = 0.1  # stripes wider than this fraction (x3) of the patch are a shadow, not a tick
    max_cropped_frac: float = 0.5  # a paper candidate cropping away more than this fraction of the image is rejected


@dataclass
class SegmentationCuttingsBlackCircleConfig:
    """Tunable parameters for segmenting cuttings laid out inside a black circle."""

    val_threshold: float = 0.16  # grayscale value above this = inside the circle (not black background)
    opening_disk: int = 7  # radius for binary opening (removes noise)
    radius_shrink: float = 0.98  # shrink the detected circle's radius by this factor before inscribing the square crop


@dataclass
class SegmentationCuttingsConfig:
    """Tunable parameters for the cuttings segmentation step."""

    downscale_factor: float = 0.25  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    pebble: SegmentationCuttingsPebbleConfig = field(default_factory=SegmentationCuttingsPebbleConfig)
    black_circle: SegmentationCuttingsBlackCircleConfig = field(default_factory=SegmentationCuttingsBlackCircleConfig)


@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation step."""

    n_workers: int = 4  # number of worker processes used to segment images in parallel
    core: SegmentationCoreConfig = field(default_factory=SegmentationCoreConfig)
    cuttings: SegmentationCuttingsConfig = field(default_factory=SegmentationCuttingsConfig)
