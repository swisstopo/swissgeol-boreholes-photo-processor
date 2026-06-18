"""Tests for the stitching module."""

from pathlib import Path

from PIL import Image

from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed
from src.stitching import GAP, PADDING, stitching


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
    return ImageMetadataProcessed(metadata=metadata, result=result)


def _full_width(num_cores: int, core_width: int) -> int:
    return PADDING + num_cores * core_width + GAP * (num_cores - 1) + PADDING


def _full_height(core_height: int) -> int:
    return PADDING + core_height + PADDING


class TestStitching:
    """Tests for the stitching function."""

    def test_single_core_produces_one_output_image(self, tmp_path):
        core = _make_processed(tmp_path, 0.0, 1.0, (100, 200))
        result = stitching([core])
        assert len(result) == 1

    def test_canvas_includes_padding_on_all_sides(self, tmp_path):
        core = _make_processed(tmp_path, 0.0, 1.0, (100, 200))
        result = stitching([core], num_cores_per_image=6)
        assert result[0].size == (_full_width(6, 100), _full_height(200))

    def test_multiple_cores_placed_side_by_side(self, tmp_path):
        cores = [_make_processed(tmp_path, float(i), float(i + 1), (100, 200)) for i in range(3)]
        result = stitching(cores, num_cores_per_image=6)
        assert len(result) == 1
        assert result[0].size == (_full_width(6, 100), _full_height(200))

    def test_full_batch_fills_one_output_image(self, tmp_path):
        cores = [_make_processed(tmp_path, float(i), float(i + 1), (100, 200)) for i in range(6)]
        result = stitching(cores, num_cores_per_image=6)
        assert len(result) == 1
        assert result[0].size == (_full_width(6, 100), _full_height(200))

    def test_overflow_both_images_have_same_size(self, tmp_path):
        cores = [_make_processed(tmp_path, float(i), float(i + 1), (100, 200)) for i in range(7)]
        result = stitching(cores, num_cores_per_image=6)
        assert len(result) == 2
        assert result[0].size == result[1].size

    def test_padding_pixels_are_black(self, tmp_path):
        core = _make_processed(tmp_path, 0.0, 1.0, (100, 200), color=(255, 0, 0))
        img = stitching([core], num_cores_per_image=6)[0]
        assert img.getpixel((0, 0)) == (0, 0, 0)  # top-left corner
        assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 0)  # bottom-right corner
        assert img.getpixel((0, img.height // 2)) == (0, 0, 0)  # left edge mid

    def test_gap_pixels_are_black(self, tmp_path):
        red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
        blue = _make_processed(tmp_path, 1.0, 2.0, (10, 10), color=(0, 0, 255))
        img = stitching([red, blue], num_cores_per_image=6)[0]
        gap_x = PADDING + 10  # one pixel into the gap between red and blue
        assert img.getpixel((gap_x, PADDING)) == (0, 0, 0)

    def test_cores_appear_in_order_left_to_right(self, tmp_path):
        red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
        blue = _make_processed(tmp_path, 1.0, 2.0, (10, 10), color=(0, 0, 255))
        img = stitching([red, blue], num_cores_per_image=6)[0]
        assert img.getpixel((PADDING, PADDING)) == (255, 0, 0)  # first core
        assert img.getpixel((PADDING + 10 + GAP, PADDING)) == (0, 0, 255)  # second core

    def test_first_core_starts_after_left_padding(self, tmp_path):
        red = _make_processed(tmp_path, 0.0, 1.0, (10, 10), color=(255, 0, 0))
        img = stitching([red], num_cores_per_image=6)[0]
        assert img.getpixel((PADDING - 1, PADDING)) == (0, 0, 0)  # last padding pixel is black
        assert img.getpixel((PADDING, PADDING)) == (255, 0, 0)  # first core pixel is red

    def test_output_size_is_resized_when_specified(self, tmp_path):
        core = _make_processed(tmp_path, 0.0, 1.0, (100, 200))
        result = stitching([core], num_cores_per_image=6, output_width=800, output_height=400)
        assert result[0].size == (800, 400)

    def test_output_width_only_keeps_natural_height(self, tmp_path):
        core = _make_processed(tmp_path, 0.0, 1.0, (100, 200))
        result = stitching([core], num_cores_per_image=6, output_width=800)
        assert result[0].width == 800
        assert result[0].height == _full_height(200)

    def test_depth_labels_do_not_change_image_size(self, tmp_path):
        core = _make_processed(tmp_path, 15.0, 16.0, (100, 200))
        result = stitching([core], num_cores_per_image=6)
        assert result[0].size == (_full_width(6, 100), _full_height(200))

    def test_depth_labels_write_into_padding_band(self, tmp_path):
        # Labels are centered over each core strip, not over the full canvas.
        core = _make_processed(tmp_path, 15.0, 16.0, (100, 200))
        img = stitching([core], num_cores_per_image=6)[0]
        cx = PADDING + 100 // 2  # center of the first (only) core strip
        top_band_pixels = [img.getpixel((cx + dx, PADDING // 2)) for dx in range(-20, 20)]
        bottom_band_pixels = [img.getpixel((cx + dx, img.height - PADDING // 2)) for dx in range(-20, 20)]
        assert any(p != (0, 0, 0) for p in top_band_pixels), "expected depth label in top padding"
        assert any(p != (0, 0, 0) for p in bottom_band_pixels), "expected depth label in bottom padding"

    def test_cores_of_different_heights_are_top_aligned(self, tmp_path):
        tall = _make_processed(tmp_path, 0.0, 1.0, (50, 300), color=(255, 0, 0))
        short = _make_processed(tmp_path, 1.0, 2.0, (50, 100), color=(0, 0, 255))
        img = stitching([tall, short], num_cores_per_image=6)[0]
        short_x = PADDING + 50 + GAP
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


class TestStitchingVisualOutput:
    """Saves stitched images to tests/output/stitching/ for manual inspection."""

    def test_save_two_output_images(self, tmp_path):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cores = [
            _make_processed(tmp_path, float(i), float(i + 1), (100, 600), color=_CORE_COLORS[i]) for i in range(7)
        ]
        results = stitching(cores, num_cores_per_image=6)
        for idx, img in enumerate(results):
            out_path = OUTPUT_DIR / f"stitched_{idx + 1}.png"
            img.save(out_path)
        print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")
