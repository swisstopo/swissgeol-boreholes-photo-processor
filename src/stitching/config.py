"""Configuration and result models for stitching pipeline."""

from dataclasses import dataclass


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
