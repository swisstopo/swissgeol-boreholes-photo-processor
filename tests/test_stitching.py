"""Tests for the stitching module."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.stitching.stitching import _resize_cores, stitching

RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Kept small so tests run fast. FONT_SIZE is well under PADDING_VERTICAL so the
# (currently hard-coded, see _draw_cores) size-100 depth-label font still fits inside
# the padding band instead of being clipped against the canvas edge.
PADDING_VERTICAL = 150
PADDING_HORIZONTAL = 60
RULER_WIDTH = 80
CORE_HEIGHT_PX = 60
CORE_HEIGHT_M = 1.0
CORE_WIDTH_RERROR = 1.5
FONT_SIZE = 40

# A (20, 100) raw crop resizes to exactly (12, 60) under the constants above:
# fx = 100 / 60 = 5/3, so width -> round(20 / (5/3)) = 12, height -> round(100 / (5/3)) = 60.
STD_RAW_SIZE = (20, 100)
STD_RESIZED_SIZE = (12, 60)


@pytest.fixture
def make_processed(tmp_path):
    def _factory(
        depth_start: float,
        depth_end: float,
        size: tuple[int, int] = STD_RAW_SIZE,
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> ImageMetadataProcessed:
        """Creates a simple ImageMetadataProcessed with a single solid-color crop of the specified size."""
        filename = f"GBC-CB50_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        Image.new("RGB", size, color=color).save(image_path)
        metadata = ImageMetadata(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )
        result = CoreSegmentResult(bounding_box=(0.0, 0.0, float(size[0]), float(size[1])))
        return ImageMetadataProcessed.from_metadata(metadata=metadata, result=result)

    return _factory


def _stitch(
    imgs: list[ImageMetadataProcessed],
    num_cores_per_image: int = 6,
    padding_vertical: int = PADDING_VERTICAL,
    padding_horizontal: int = PADDING_HORIZONTAL,
    ruler_width: int = RULER_WIDTH,
    core_height_px: int = CORE_HEIGHT_PX,
    core_height_m: float = CORE_HEIGHT_M,
    core_width_rerror: float = CORE_WIDTH_RERROR,
    font_size: int = FONT_SIZE,
):
    """Calls stitching() with this module's small, fast test defaults, allowing per-test overrides."""
    return stitching(
        imgs,
        num_cores_per_image=num_cores_per_image,
        padding_vertical=padding_vertical,
        padding_horizontal=padding_horizontal,
        ruler_width=ruler_width,
        core_height_px=core_height_px,
        core_height_m=core_height_m,
        core_width_rerror=core_width_rerror,
        font_size=font_size,
    )


def _predict_resized_sizes(
    sizes: list[tuple[int, int]],
    core_height_px: float,
    core_height_m: float = 1.0,
    core_width_rerror: float = 1.5,
) -> list[tuple[int, int]]:
    """Mirrors _resize_cores' scale-selection/outlier-clamping formula, for use as a test oracle."""
    widths = np.array([w for w, _ in sizes], dtype=float)
    heights = np.array([h for _, h in sizes], dtype=float)
    error_width = np.abs(widths) / np.median(widths)
    error_height = np.abs(heights) / np.median(heights)
    id_ref = int(np.argmin(1 - error_width * error_height))
    ref_fx = heights[id_ref] / (core_height_px * core_height_m)

    result = []
    for w, h in sizes:
        fx = h / (core_height_px * core_height_m)
        if w / fx > core_width_rerror * widths[id_ref] / ref_fx:
            fx = ref_fx
        result.append((round(w / fx), round(h / fx)))
    return result


def _predict_canvas_size(
    resized_sizes: list[tuple[int, int]],
    padding_horizontal: int,
    padding_vertical: int,
    ruler_width: int,
    core_height_px: int,
) -> tuple[int, int]:
    """Mirrors stitching_batch's canvas-size formula."""
    n = len(resized_sizes)
    cores_width = sum(w for w, _ in resized_sizes)
    canvas_width = (3 + n) * padding_horizontal + 2 * ruler_width + cores_width
    canvas_height = 5 * padding_vertical + core_height_px
    return canvas_width, canvas_height


def _predict_core_x_positions(
    resized_sizes: list[tuple[int, int]], padding_horizontal: int, ruler_width: int
) -> list[int]:
    """Mirrors stitching_batch's core placement formula (left-aligned, in order, spaced by padding_horizontal)."""
    x = 2 * padding_horizontal + ruler_width
    xs = []
    for w, _ in resized_sizes:
        xs.append(x)
        x += w + padding_horizontal
    return xs


# ---- _resize_cores -----------------------------------------------------------------
# Note: _resize_cores scales purely from each crop's raw pixel size vs. a reference
# core in the same batch. It does NOT use depth_start/depth_end, so core height is
# currently independent of the core's physical depth extent.


def test_resize_cores_normalizes_same_aspect_cores_to_a_common_height():
    """Cores that share an aspect ratio are all resized to the same target height."""
    cores = [Image.new("RGB", STD_RAW_SIZE) for _ in range(3)]
    resized = _resize_cores(cores, core_height_px=60, core_height_m=1.0, core_width_rerror=1.5)
    assert [(c.width, c.height) for c in resized] == [STD_RESIZED_SIZE] * 3


def test_resize_cores_matches_predicted_sizes_for_a_mixed_batch():
    """_resize_cores' output matches an independent replica of its scale/outlier formula."""
    sizes = [(20, 100), (20, 300), (40, 120)]
    cores = [Image.new("RGB", size) for size in sizes]
    resized = _resize_cores(cores, core_height_px=60, core_height_m=1.0, core_width_rerror=1.2)
    expected = _predict_resized_sizes(sizes, core_height_px=60, core_height_m=1.0, core_width_rerror=1.2)
    assert [(c.width, c.height) for c in resized] == expected


def test_resize_cores_outlier_is_rescaled_using_the_reference_core():
    """A core whose own scale factor would make it disproportionately wide is instead rescaled.

    The rescale uses the reference core's factor, which also changes its height.
    """
    # core0: (20, 100), normal aspect. core1: (20, 300), much taller at the same width,
    # so its error_height dominates and it becomes the reference core.
    sizes = [(20, 100), (20, 300)]
    cores = [Image.new("RGB", size) for size in sizes]
    resized = _resize_cores(cores, core_height_px=60, core_height_m=1.0, core_width_rerror=1.2)
    expected = _predict_resized_sizes(sizes, core_height_px=60, core_height_m=1.0, core_width_rerror=1.2)
    assert [(c.width, c.height) for c in resized] == expected
    # core0's own scale would give it height 60 (like STD_RESIZED_SIZE); confirm it was
    # actually pulled onto the reference's scale instead, i.e. the outlier path fired.
    assert resized[0].height != 60


# ---- stitching / stitching_batch ----------------------------------------------------


def test_output_count_and_canvas_dimensions(make_processed):
    """Stitching chunks by num_cores_per_image, and each chunk's canvas is sized from its own core count."""
    cores = [make_processed(float(i), float(i + 1)) for i in range(7)]
    results = list(_stitch(cores, num_cores_per_image=6))

    assert len(results) == 2
    assert results[0].size == _predict_canvas_size(
        [STD_RESIZED_SIZE] * 6, PADDING_HORIZONTAL, PADDING_VERTICAL, RULER_WIDTH, CORE_HEIGHT_PX
    )
    assert results[1].size == _predict_canvas_size(
        [STD_RESIZED_SIZE] * 1, PADDING_HORIZONTAL, PADDING_VERTICAL, RULER_WIDTH, CORE_HEIGHT_PX
    )


def test_padding_pixels_are_black(make_processed):
    """Padding pixels around the image are black, not white or some other color."""
    core = make_processed(0.0, 1.0, color=RED)
    img = next(_stitch([core], num_cores_per_image=6))
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left margin, before the ruler


def test_cores_appear_in_order_left_to_right(make_processed):
    """Cores appear in the output in the same order as the input list, from left to right."""
    red = make_processed(0.0, 1.0, color=RED)
    blue = make_processed(1.0, 2.0, color=BLUE)
    img = next(_stitch([red, blue], num_cores_per_image=6))
    xs = _predict_core_x_positions([STD_RESIZED_SIZE, STD_RESIZED_SIZE], PADDING_HORIZONTAL, RULER_WIDTH)
    y = 3 * PADDING_VERTICAL + 5
    assert img.getpixel((xs[0] + 2, y)) == RED
    assert img.getpixel((xs[1] + 2, y)) == BLUE


def test_gap_between_cores_is_black(make_processed):
    """The padding_horizontal-wide gap between adjacent cores is black."""
    red = make_processed(0.0, 1.0, color=RED)
    blue = make_processed(1.0, 2.0, color=BLUE)
    img = next(_stitch([red, blue], num_cores_per_image=6))
    xs = _predict_core_x_positions([STD_RESIZED_SIZE, STD_RESIZED_SIZE], PADDING_HORIZONTAL, RULER_WIDTH)
    gap_x = xs[0] + STD_RESIZED_SIZE[0] + PADDING_HORIZONTAL // 2
    y = 3 * PADDING_VERTICAL + 5
    assert img.getpixel((gap_x, y)) == (0, 0, 0)


def test_cores_are_top_aligned(make_processed):
    """Cores are pasted starting at the top of the content band, not centred within it."""
    core = make_processed(0.0, 1.0, color=RED)
    img = next(_stitch([core], num_cores_per_image=6))
    x = _predict_core_x_positions([STD_RESIZED_SIZE], PADDING_HORIZONTAL, RULER_WIDTH)[0]
    y_top = 3 * PADDING_VERTICAL
    assert img.getpixel((x + 2, y_top)) == RED
    assert img.getpixel((x + 2, y_top - 1)) != RED


def test_depth_labels_are_drawn_above_and_below_each_core(make_processed):
    """Depth labels are drawn in the padding bands above and below the core content."""
    core = make_processed(15.0, 16.0, color=RED)
    img = next(_stitch([core], num_cores_per_image=6))
    y_min = 3 * PADDING_VERTICAL
    above_y = y_min - PADDING_VERTICAL
    below_y = y_min + STD_RESIZED_SIZE[1] + PADDING_VERTICAL
    above_row = [img.getpixel((x, above_y)) for x in range(img.width)]
    below_row = [img.getpixel((x, below_y)) for x in range(img.width)]
    assert any(p != (0, 0, 0) for p in above_row), "expected a depth label above the core"
    assert any(p != (0, 0, 0) for p in below_row), "expected a depth label below the core"


def test_borehole_label_is_drawn(make_processed):
    """The borehole ID is drawn somewhere in the top label row, left of the core content."""
    core = make_processed(0.0, 1.0)
    img = next(_stitch([core], num_cores_per_image=6))
    row = [img.getpixel((x, PADDING_VERTICAL)) for x in range(img.width)]
    assert any(p != (0, 0, 0) for p in row)


def test_rulers_are_drawn_on_both_sides(make_processed):
    """Both the left and right depth rulers draw at least one tick."""
    core = make_processed(0.0, 1.0)
    img = next(_stitch([core], num_cores_per_image=6))
    y_first_tick = 3 * PADDING_VERTICAL

    left_row = [img.getpixel((x, y_first_tick)) for x in range(PADDING_HORIZONTAL, PADDING_HORIZONTAL + RULER_WIDTH)]
    assert any(p != (0, 0, 0) for p in left_row), "expected a tick on the left ruler"

    right_ruler_x0 = 3 * PADDING_HORIZONTAL + RULER_WIDTH + STD_RESIZED_SIZE[0]
    right_row = [img.getpixel((x, y_first_tick)) for x in range(right_ruler_x0, right_ruler_x0 + RULER_WIDTH)]
    assert any(p != (0, 0, 0) for p in right_row), "expected a tick on the right ruler"


def test_outlier_core_width_matches_the_reference_core(make_processed):
    """An outlier core (see _resize_cores) is placed and sized consistently with the resize step."""
    normal = make_processed(0.0, 1.0, size=(20, 100), color=RED)
    outlier = make_processed(1.0, 4.0, size=(20, 300), color=BLUE)
    core_width_rerror = 1.2
    img = next(_stitch([normal, outlier], num_cores_per_image=6, core_width_rerror=core_width_rerror))

    resized = _predict_resized_sizes(
        [(20, 100), (20, 300)],
        core_height_px=CORE_HEIGHT_PX,
        core_height_m=CORE_HEIGHT_M,
        core_width_rerror=core_width_rerror,
    )
    xs = _predict_core_x_positions(resized, PADDING_HORIZONTAL, RULER_WIDTH)
    y = 3 * PADDING_VERTICAL + 2

    assert img.getpixel((xs[0] + 1, y)) == RED
    assert img.getpixel((xs[1] + 1, y)) == BLUE
    assert img.size == _predict_canvas_size(resized, PADDING_HORIZONTAL, PADDING_VERTICAL, RULER_WIDTH, CORE_HEIGHT_PX)


OUTPUT_DIR = Path(__file__).parent / "output" / "stitching"

_CORE_COLORS = [
    (220, 50, 50),  # red
    (220, 140, 50),  # orange
    (200, 200, 50),  # yellow
    (50, 180, 50),  # green
    (50, 140, 220),  # blue
    (140, 50, 220),  # purple
    (180, 180, 180),  # grey  — overflow core on second image
]


# @pytest.mark.skip(reason="visual inspection only — run manually")
def test_save_two_output_images(make_processed):
    """Creates two output images with 6 cores in the first and 1 core in the second, for visual inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cores = [make_processed(float(i), float(i + 1), color=_CORE_COLORS[i]) for i in range(7)]
    results = _stitch(cores, num_cores_per_image=6)
    for idx, img in enumerate(results):
        out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
        img.save(out_path)
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
