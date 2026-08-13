"""Configuration and result models for stitching pipeline."""

from dataclasses import dataclass, field


@dataclass
class CoreStitchingConfig:
    """Tunable parameters for stitching core crops into ruler-labeled sheets."""

    font_size: int = 100  # font size (px) used for borehole ID, depth labels, and ruler tick labels
    max_core_height: int = 10000  # cap on each core crop's resized height (pixels) and the canvas row height
    max_core_width: int = 1200  # cap on each core crop's resized width (pixels) and the per-core column width
    num_cores_per_image: int = 6  # cores placed side by side per output sheet
    padding_horizontal: int = 150  # left/right border width in pixels
    padding_vertical: int = 200  # top/bottom border height in pixels
    ruler_width: int = 300  # width in pixels of each of the two depth rulers (left/right of the cores)


@dataclass
class CuttingsStitchingConfig:
    """Tunable parameters for arranging cuttings images into a fixed grid."""

    annotation_gap: int = 20  # gap in pixels between a cutting's image and its depth label
    annotation_width: int = 130  # width in pixels reserved at the right of each cell for the depth label
    font_size: int = 40  # font size (px) used for the depth label next to each cutting
    output_height: int = 2896  # total output canvas height (pixels), split evenly across the rows
    output_width: int = 2048  # total output canvas width (pixels), split evenly across the columns
    num_cuttings_columns: int = 3  # cuttings placed side by side per output sheet
    num_cuttings_rows: int = 15  # cuttings stacked per column per output sheet
    padding_cuttings: int = 20  # gap in pixels between neighboring cells, carved out of the output dimensions
    padding_horizontal: int = 170  # left/right border width in pixels
    padding_vertical: int = 150  # top/bottom border height in pixels


@dataclass
class StitchingConfig:
    """Tunable parameters for the stitching step."""

    n_workers: int = 4  # number of worker processes used to segment images in parallel
    web_output_quality: int = 95  # output JPG quality/compression tradeoff, in the range 0–100
    web_downscale_factor: float = 0.25  # downscale factor applied to web image output
    core: CoreStitchingConfig = field(default_factory=CoreStitchingConfig)
    cuttings: CuttingsStitchingConfig = field(default_factory=CuttingsStitchingConfig)
