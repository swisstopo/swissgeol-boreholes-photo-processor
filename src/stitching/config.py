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

    output_height: int = 2896  # total output canvas height (pixels), split evenly across the rows
    output_width: int = 2048  # total output canvas width (pixels), split evenly across the columns
    num_cuttings_columns: int = 3  # cuttings placed side by side per output sheet
    num_cuttings_rows: int = 20  # cuttings stacked per column per output sheet
    spacing: int = 20  # gap in pixels between neighboring cells, carved out of the output dimensions


@dataclass
class StitchingConfig:
    """Tunable parameters for the stitching step."""

    core: CoreStitchingConfig = field(default_factory=CoreStitchingConfig)
    cuttings: CuttingsStitchingConfig = field(default_factory=CuttingsStitchingConfig)
