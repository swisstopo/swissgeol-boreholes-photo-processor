"""Tests for the stitching module."""

from pathlib import Path

import pytest
from PIL import Image

from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.stitching import OUTPUT_HEIGHT, OUTPUT_WIDTH, PADDING, stitching


def _make_processed(
    tmp_path: Path,
    depth_start: float,
    depth_end: float,
    size: tuple[int, int],
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


def _derived_gap(crop_width: int, num_cores_per_image: int = 6, output_width: int = OUTPUT_WIDTH) -> int:
    full_batch_gaps = num_cores_per_image - 1
    if full_batch_gaps == 0:
        return 0
    return max(0, (output_width - 2 * PADDING - crop_width * num_cores_per_image) // full_batch_gaps)


@pytest.mark.parametrize(
    "num_cores, expected_count",
    [
        (1, 1),
        (3, 1),
        (6, 1),
        (7, 2),
    ],
)
def test_output_count_and_dimensions(tmp_path, num_cores, expected_count):
    cores = [_make_processed(tmp_path, float(i), float(i + 1), (100, 200)) for i in range(num_cores)]
    result = list(stitching(cores, num_cores_per_image=6))
    assert len(result) == expected_count
    assert all(img.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT) for img in result)


def test_padding_pixels_are_black(tmp_path):
    core = _make_processed(tmp_path, 0.0, 1.0, (100, 200), color=(255, 0, 0))
    img = next(stitching([core], num_cores_per_image=6))
    assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
    assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left edge mid


def test_gap_pixels_are_black(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
    blue = _make_processed(tmp_path, 1.0, 2.0, (10, 10), color=(0, 0, 255))
    img = next(stitching([red, blue], num_cores_per_image=6))
    gap_x = PADDING + 10  # one pixel into the gap between red and blue
    assert img.getpixel((gap_x, PADDING)) == (0, 0, 0)


def test_cores_appear_in_order_left_to_right(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
    blue = _make_processed(tmp_path, 1.0, 2.0, (10, 10), color=(0, 0, 255))
    img = next(stitching([red, blue], num_cores_per_image=6))
    gap = _derived_gap(crop_width=10, num_cores_per_image=6)
    assert img.getpixel((PADDING, PADDING)) == (255, 0, 0)  # first core
    assert img.getpixel((PADDING + 10 + gap, PADDING)) == (0, 0, 255)  # second core


def test_first_core_starts_after_left_padding(tmp_path):
    red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
    img = next(stitching([red], num_cores_per_image=6))
    assert img.getpixel((PADDING - 1, PADDING)) == (0, 0, 0)  # last padding pixel is black
    assert img.getpixel((PADDING, PADDING)) == (255, 0, 0)  # first core pixel is red


def test_custom_output_dimensions_create_canvas_at_that_size(tmp_path):
    core = _make_processed(tmp_path, 0.0, 1.0, (100, 200))
    img = next(stitching([core], num_cores_per_image=6, output_width=800, output_height=400))
    assert img.size == (800, 400)


def test_depth_labels_write_into_padding_band(tmp_path):
    core = _make_processed(tmp_path, 15.0, 16.0, (100, 200))
    img = next(stitching([core], num_cores_per_image=6))
    cx = PADDING + 100 // 2  # center of the first (only) core strip
    top_band_pixels = [img.getpixel((cx + dx, PADDING // 2)) for dx in range(-20, 20)]
    bottom_band_pixels = [img.getpixel((cx + dx, img.height - PADDING // 2)) for dx in range(-20, 20)]
    assert any(p != (0, 0, 0) for p in top_band_pixels), "expected depth label in top padding"
    assert any(p != (0, 0, 0) for p in bottom_band_pixels), "expected depth label in bottom padding"


def test_cores_of_different_heights_are_top_aligned(tmp_path):
    tall = _make_processed(tmp_path, 0.0, 1.0, (50, 300), color=(255, 0, 0))
    short = _make_processed(tmp_path, 1.0, 2.0, (50, 100), color=(0, 0, 255))
    img = next(stitching([tall, short], num_cores_per_image=6))
    gap = _derived_gap(crop_width=50, num_cores_per_image=6)
    short_x = PADDING + 50 + gap
    assert img.getpixel((short_x, PADDING)) == (0, 0, 255)  # short core top
    assert img.getpixel((short_x, PADDING + 100)) == (0, 0, 0)  # below short core is black


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


def test_save_two_output_images(tmp_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cores = [_make_processed(tmp_path, float(i), float(i + 1), (100, 600), color=_CORE_COLORS[i]) for i in range(7)]
    results = stitching(cores, num_cores_per_image=4)
    for idx, img in enumerate(results):
        out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
        img.save(out_path)
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
