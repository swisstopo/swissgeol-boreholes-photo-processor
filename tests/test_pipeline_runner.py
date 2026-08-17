"""Tests for the pipeline_runner module."""

from pathlib import Path
from typing import Any

from PIL import Image

from src.config import PipelineConfig
from src.pipeline_runner import CuttingsPipelineRunner, PipelineRunner


class _FakeRunner(PipelineRunner[Any, Any, Any]):
    """Minimal concrete runner for exercising the base class's run() template in isolation."""

    def _collect(self, input_dir: Path, with_mlflow: bool) -> list[Any]:
        return []

    def _segment(self, imgs_metadata, config, with_mlflow, debug, cache, cut_type="black_circle") -> list[Any]:
        return []

    def _collate_stitch(self, imgs, config) -> list[Any]:
        return []

    def _batch_stitch(self, batch: Any, config) -> Image.Image:
        return Image.new("RGB", (1, 1))


def test_run_with_no_images_creates_empty_output_dir_without_crashing(tmp_path):
    """run() handles the empty-collection/empty-stitch path (the idx=-1 guard) without raising."""
    output_dir = tmp_path / "output"

    _FakeRunner().run(input_dir=tmp_path, output_dir=output_dir, config=PipelineConfig())

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_cuttings_pipeline_runner_produces_output_for_depth_photos(tmp_path):
    """CuttingsPipelineRunner.run() end-to-end on a couple of depth-labeled cuttings photos."""
    input_dir = tmp_path / "GES-F-1"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    Image.new("RGB", (60, 80), color=(128, 128, 128)).save(input_dir / "10m_00.jpg")
    Image.new("RGB", (60, 80), color=(128, 128, 128)).save(input_dir / "20m_00.jpg")

    CuttingsPipelineRunner().run(input_dir=input_dir, output_dir=output_dir, config=PipelineConfig())

    assert (output_dir / "GES-F-1_001.jpg").exists()
    assert (output_dir / "GES-F-1_001.tif").exists()
