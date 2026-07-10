"""Helper functions for stitching (currently unused, reserved for shared utilities)."""

import numpy as np
from PIL import Image


def _resize_cores(
    cores: list[Image.Image],
    core_height_px: float,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
) -> list[Image.Image]:
    """Resize cores to a common pixel scale, clamping outlier widths to the reference core.

    Args:
        cores (list[Image.Image]): Raw core crops to resize.
        core_height_px (float): Pixel budget for a core spanning core_height_m metres.
        core_height_m (float): Depth extent, in metres, that core_height_px pixels represents.
        core_width_rerror (float): Maximum allowed width ratio vs. the reference core before
            a core is treated as an outlier and rescaled using the reference core's scale factor.

    Returns:
        list[Image.Image]: Cores resized to a consistent pixel scale.
    """
    # Pick the core closest to the batch median in both width and height as the reference
    widths = np.array([core.width for core in cores])
    heights = np.array([core.height for core in cores])

    error_width = np.abs(1 - widths / np.median(widths))
    error_height = np.abs(1 - heights / np.median(heights))
    id_ref = np.argmin(error_width + error_height)

    ref_fx = cores[id_ref].height / (core_height_px * core_height_m)

    cores_resized: list[Image.Image] = []
    for core in cores:
        # Core is within acceptable range
        fx = core.height / (core_height_px * core_height_m)

        if core.width / fx > core_width_rerror * cores[id_ref].width / ref_fx:
            fx = ref_fx

        cores_resized.append(core.resize((round(core.width / fx), round(core.height / fx)), Image.Resampling.LANCZOS))

    return cores_resized
