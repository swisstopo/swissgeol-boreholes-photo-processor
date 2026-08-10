"""Tests for the pipeline_runner module."""

from collections.abc import Generator
from pathlib import Path

from PIL import Image

from src.config import PipelineConfig
from src.models import ImageMetadata
from src.pipeline_runner import CorePipelineRunner, CuttingsPipelineRunner, PipelineRunner


class _FakeRunner(PipelineRunner[ImageMetadata, ImageMetadata]):
    """Minimal concrete runner for exercising the base class's run() template in isolation."""

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[ImageMetadata]:
        return []

    def _segment(self, imgs_metadata, config, with_mlflow, debug) -> list[ImageMetadata]:
        return []

    def _stitch(self, imgs, config) -> Generator[Image.Image, None, None]:
        yield from ()


def test_run_with_no_images_creates_empty_output_dir_without_crashing(tmp_path):
    """run() handles the empty-collection/empty-stitch path (the idx=-1 guard) without raising."""
    output_dir = tmp_path / "output"

    _FakeRunner().run(input_dir=tmp_path, output_dir=output_dir, config=PipelineConfig())

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_core_pipeline_runner_produces_output_for_example_photo(tmp_path):
    """CorePipelineRunner.run() end-to-end on the real example core/tray/ruler photo."""
    input_dir = tmp_path / "EX-EX"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    Image.open("examples/EX-EX_0001.00-002.00.jpg").save(input_dir / "EX-EX_0001.00-0002.00_vd_p.tif")

    CorePipelineRunner().run(input_dir=input_dir, output_dir=output_dir, config=PipelineConfig())

    assert (output_dir / "EX-EX_001.png").exists()
    assert (output_dir / "EX-EX_001.tif").exists()


def test_cuttings_pipeline_runner_produces_output_for_depth_photos(tmp_path):
    """CuttingsPipelineRunner.run() end-to-end on a couple of depth-labeled cuttings photos."""
    input_dir = tmp_path / "GES-F-1"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    Image.new("RGB", (60, 80), color=(128, 128, 128)).save(input_dir / "10m_00.jpg")
    Image.new("RGB", (60, 80), color=(128, 128, 128)).save(input_dir / "20m_00.jpg")

    CuttingsPipelineRunner().run(input_dir=input_dir, output_dir=output_dir, config=PipelineConfig())

    assert (output_dir / "GES-F-1_001.png").exists()
    assert (output_dir / "GES-F-1_001.tif").exists()
