"""Tests for the stitching module."""

from pathlib import Path

import pytest
from PIL import Image

from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.stitching import (
    MAX_CORE_LENGTH_M,
    NUM_CORES_PER_IMAGE,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PADDING_HORIZONTAL,
    PADDING_VERTICAL,
    stitching,
)

_CORE_STRIP_HEIGHT = OUTPUT_HEIGHT - 2 * PADDING_VERTICAL

# Standard test-crop size. With depth extent 1.0 it resizes to (136, 1070)
# — aspect-preserving at _CORE_STRIP_HEIGHT=1070.
_STD_CROP_SIZE = (14, 110)


def _make_processed(
    tmp_path: Path,
    depth_start: float,
    depth_end: float,
    size: tuple[int, int] = _STD_CROP_SIZE,
    color: tuple[int, int, int] = (128, 128, 128),
) -> ImageMetadataProcessed:
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


def _resized_width(orig_w: int, orig_h: int, depth_start: float, depth_end: float) -> int:
    """Expected crop width after _resize_core (mirrors its logic)."""
    target_h = max(1, round((depth_end - depth_start) / MAX_CORE_LENGTH_M * _CORE_STRIP_HEIGHT))
    return max(1, round(target_h * orig_w / orig_h))


def _resized_height(depth_start: float, depth_end: float) -> int:
    """Expected crop height after _resize_core."""
    return max(1, round((depth_end - depth_start) / MAX_CORE_LENGTH_M * _CORE_STRIP_HEIGHT))


def _derived_gap(crop_widths: list[int], num_cores: int, output_width: int = OUTPUT_WIDTH) -> int:
    if num_cores <= 1:
        return 0
    return max(0, (output_width - 2 * PADDING_HORIZONTAL - sum(crop_widths)) // (num_cores - 1))


def _x_start(crop_widths: list[int], gap: int, output_width: int = OUTPUT_WIDTH) -> int:
    total_content_width = sum(crop_widths) + gap * max(0, len(crop_widths) - 1)
    return (output_width - total_content_width) // 2


def _std_crop_widths(n: int, depth_extent: float = 1.0) -> list[int]:
    """Widths of n standard crops after resize (all the same when depth_extent is uniform)."""
    w = _resized_width(_STD_CROP_SIZE[0], _STD_CROP_SIZE[1], 0.0, depth_extent)
    return [w] * n


@pytest.mark.parametrize(
    "num_cores, expected_count",
    [
        # num_cores = total input cores; chunk size is always NUM_CORES_PER_IMAGE=6
        (1, 1),
        (3, 1),
        (6, 1),
        (7, 2),  # 7 cores split into chunks of 6 → 2 output images
    ],
)
def test_output_count_and_dimensions(tmp_path, num_cores, expected_count):
    cores = [_make_processed(tmp_path, float(i), float(i + 1)) for i in range(num_cores)]
    result = list(stitching(cores, num_cores_per_image=NUM_CORES_PER_IMAGE))
    assert len(result) == expected_count
    assert all(img.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT) for img in result)


def test_padding_pixels_are_black(tmp_path):
    core = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    img = next(stitching([core], num_cores_per_image=NUM_CORES_PER_IMAGE))
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left edge mid


def test_gap_pixels_are_black(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    blue = _make_processed(tmp_path, 1.0, 2.0, color=(0, 0, 255))
    img = next(stitching([red, blue], num_cores_per_image=NUM_CORES_PER_IMAGE))
    all_widths = _std_crop_widths(NUM_CORES_PER_IMAGE)
    gap = _derived_gap(all_widths, NUM_CORES_PER_IMAGE)
    assert gap > 0, "gap must be positive for this test to be meaningful"
    x0 = _x_start(all_widths, gap)
    gap_x = x0 + all_widths[0]  # first pixel after the first crop slot
    assert img.getpixel((gap_x, PADDING_VERTICAL)) == (0, 0, 0)


def test_cores_appear_in_order_left_to_right(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    blue = _make_processed(tmp_path, 1.0, 2.0, color=(0, 0, 255))
    img = next(stitching([red, blue], num_cores_per_image=NUM_CORES_PER_IMAGE))
    all_widths = _std_crop_widths(NUM_CORES_PER_IMAGE)
    gap = _derived_gap(all_widths, NUM_CORES_PER_IMAGE)
    x0 = _x_start(all_widths, gap)
    assert img.getpixel((x0, PADDING_VERTICAL)) == (255, 0, 0)  # first core
    assert img.getpixel((x0 + all_widths[0] + gap, PADDING_VERTICAL)) == (0, 0, 255)  # second core


def test_first_core_starts_after_left_padding(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    img = next(stitching([red], num_cores_per_image=NUM_CORES_PER_IMAGE))
    all_widths = _std_crop_widths(NUM_CORES_PER_IMAGE)
    gap = _derived_gap(all_widths, NUM_CORES_PER_IMAGE)
    x0 = _x_start(all_widths, gap)
    assert img.getpixel((x0 - 1, PADDING_VERTICAL)) == (0, 0, 0)  # pixel before first core is black
    assert img.getpixel((x0, PADDING_VERTICAL)) == (255, 0, 0)  # first core pixel is red


def test_custom_output_dimensions_create_canvas_at_that_size(tmp_path):
    core = _make_processed(tmp_path, 0.0, 1.0)
    img = next(stitching([core], num_cores_per_image=NUM_CORES_PER_IMAGE, output_width=800, output_height=400))
    assert img.size == (800, 400)


def test_depth_labels_write_into_padding_band(tmp_path):
    core = _make_processed(tmp_path, 15.0, 16.0)
    img = next(stitching([core], num_cores_per_image=NUM_CORES_PER_IMAGE))
    all_widths = _std_crop_widths(NUM_CORES_PER_IMAGE)
    gap = _derived_gap(all_widths, NUM_CORES_PER_IMAGE)
    x0 = _x_start(all_widths, gap)
    cx = x0 + all_widths[0] // 2  # center of the first core strip
    top_band_pixels = [img.getpixel((cx + dx, PADDING_VERTICAL * 3 // 4)) for dx in range(-20, 20)]
    bottom_band_pixels = [img.getpixel((cx + dx, img.height - PADDING_VERTICAL // 2)) for dx in range(-20, 20)]
    assert any(p != (0, 0, 0) for p in top_band_pixels), "expected depth label in top padding"
    assert any(p != (0, 0, 0) for p in bottom_band_pixels), "expected depth label in bottom padding"


def test_cores_of_different_heights_are_top_aligned(tmp_path):
    # Height is controlled by depth extent: 1.0 m → full height, 0.5 m → half height.
    tall = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    short = _make_processed(tmp_path, 1.0, 1.5, color=(0, 0, 255))
    img = next(stitching([tall, short], num_cores_per_image=NUM_CORES_PER_IMAGE))
    w_tall = _resized_width(*_STD_CROP_SIZE, 0.0, 1.0)
    w_short = _resized_width(*_STD_CROP_SIZE, 1.0, 1.5)
    h_short = _resized_height(1.0, 1.5)
    avg_placeholder_w = round((w_tall + w_short) / 2)
    all_widths = [w_tall, w_short] + [avg_placeholder_w] * (NUM_CORES_PER_IMAGE - 2)
    gap = _derived_gap(all_widths, NUM_CORES_PER_IMAGE)
    x0 = _x_start(all_widths, gap)
    short_x = x0 + w_tall + gap
    assert img.getpixel((short_x, PADDING_VERTICAL)) == (0, 0, 255)  # short core top is blue
    assert img.getpixel((short_x, PADDING_VERTICAL + h_short + 1)) == (0, 0, 0)  # below short core is black


def test_partial_chunk_padded_with_black_placeholders(tmp_path):
    """A partial last chunk fills empty slots with black boxes so layout matches a full chunk."""
    red = _make_processed(tmp_path, 0.0, 1.0, color=(255, 0, 0))
    img = next(stitching([red], num_cores_per_image=2))
    w = _resized_width(*_STD_CROP_SIZE, 0.0, 1.0)
    all_widths = [w, w]  # 1 real + 1 placeholder (same avg width)
    gap = _derived_gap(all_widths, num_cores=2)
    x0 = _x_start(all_widths, gap)
    assert img.getpixel((x0, PADDING_VERTICAL)) == (255, 0, 0)  # real core is red
    placeholder_cx = x0 + w + gap + w // 2
    assert img.getpixel((placeholder_cx, PADDING_VERTICAL)) == (0, 0, 0)  # placeholder is black


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
def test_save_two_output_images(tmp_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cores = [_make_processed(tmp_path, float(i), float(i + 1), color=_CORE_COLORS[i]) for i in range(7)]
    results = stitching(cores, num_cores_per_image=NUM_CORES_PER_IMAGE)
    for idx, img in enumerate(results):
        out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
        img.save(out_path)
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
