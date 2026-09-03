"""Configuration and result models for segmentation pipeline."""

from dataclasses import dataclass, field
from typing import Literal


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
    max_cropped_frac: float = 0.5  # a paper candidate cropping away more than this fraction of the image is rejected


@dataclass
class SegmentationCuttingsPebbleGroupConfig:
    """Tunable parameters for estimating a shared paper-sheet position across a group of same-shaped pebble images.

    Averaging many same-shape images washes the cuttings (different rock every shot) into a
    formless blur, but the paper survives sharp wherever the camera framing places it
    consistently -- so this detects the paper via cross-image standard deviation (low = stayed
    in the same place across the group) rather than per-frame brightness thresholding.
    """

    n_min_group: int = 10  # minimum images in a shape group required to attempt a shared estimate
    seed: int = 0  # seed for randomly sampling images within each group
    downscale_factor: float = 0.25  # scale images by this factor before estimating (< 1.0 speeds up morphology)
    std_percentile: float = 12  # pixels below this percentile of cross-image std = "consistently in the same spot"
    val_threshold: float = 0.55  # brightness (HSV), on the mean image, above this = bright enough to be paper
    sat_threshold: float = 0.20  # saturation (HSV), on the mean image, below this = colorless enough
    closing_disk: int = 25  # radius for binary closing that fills the black stripes punched into the paper mask
    min_extent: float = 0.6  # minimum bounding-box fill fraction for a paper candidate
    min_solidity: float = 0.75  # minimum convexity for a paper candidate
    min_area_frac: float = 0.003  # minimum paper candidate area, as a fraction of the group image area
    max_area_frac: float = 0.35  # maximum paper candidate area, as a fraction of the group image area
    edge_margin: float = 0.05  # fraction of the relevant dimension a candidate must reach to count as edge-anchored
    max_cropped_frac: float = 0.5  # a paper candidate cropping away more than this fraction of the image is rejected


@dataclass
class SegmentationCuttingsBlackCircleConfig:
    """Tunable parameters for segmenting cuttings laid out inside a black circle."""

    val_threshold: float = 0.16  # grayscale value above this = inside the circle (not black background)
    opening_disk: int = 7  # radius for binary opening (removes noise)
    radius_shrink: float = 0.98  # shrink the detected circle's radius by this factor before inscribing the square crop
    min_area_frac: float = 0.01  # a detected region below this fraction of the image area is noise, not a real circle


@dataclass
class SegmentationCuttingsTrayConfig:
    """Tunable parameters for segmenting cuttings laid out in an open tray, via edge-density quantile bbox."""

    coverage: float = 0.95  # fraction of mask pixels the box should contain (jointly, both axes)
    square: bool = False  # force a square box
    work: int = 800  # working resolution the image is resized to before computing texture energy
    open_radius: int = 3  # radius for opening (drops thin bridges/specks before picking the main component)
    erosion_radius: int = 2  # radius for eroding the main component before the bbox, to trim the residual tray border
    min_area_frac: float = 0.01  # a detected component below this fraction of the work area is noise, not a real pile


@dataclass
class SegmentationCuttingsConfig:
    """Tunable parameters for the cuttings segmentation step."""

    downscale_factor: float = 0.25  # scale images by this factor before segmenting (< 1.0 speeds up morphology)
    dedup_keep: Literal["first", "last"] = "first"  # which image to keep when multiple share the same depth
    min_crop_px: int = 20  # a real cuttings region should never be this thin; backstop against any segmenter's bugs
    crop_size_cv_warn_threshold: float = 0.3  # coefficient of variation above this is unusually inconsistent
    pebble: SegmentationCuttingsPebbleConfig = field(default_factory=SegmentationCuttingsPebbleConfig)
    pebble_group: SegmentationCuttingsPebbleGroupConfig = field(default_factory=SegmentationCuttingsPebbleGroupConfig)
    black_circle: SegmentationCuttingsBlackCircleConfig = field(default_factory=SegmentationCuttingsBlackCircleConfig)
    tray: SegmentationCuttingsTrayConfig = field(default_factory=SegmentationCuttingsTrayConfig)


@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation step."""

    n_workers: int = 4  # number of worker processes used to segment images in parallel
    core: SegmentationCoreConfig = field(default_factory=SegmentationCoreConfig)
    cuttings: SegmentationCuttingsConfig = field(default_factory=SegmentationCuttingsConfig)
