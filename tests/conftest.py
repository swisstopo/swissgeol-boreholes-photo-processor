"""Shared fixtures for tests."""

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from src.models import ImageMetadataCores

_IMG_SIZE = (1200, 800)
_BACKGROUND_COLOR = (20, 20, 20)  # dark, unsaturated — stands in for the tray backdrop


@pytest.fixture
def draw_core():
    def _factory(canvas: Image.Image, bbox: tuple[int, int, int, int]):
        x0, y0, x1, y1 = bbox
        with Image.open(Path("examples/EX-EX_0001.00-002.00.jpg")) as img:
            img = img.crop((100, 1300, 9800, 2000))
            resized_img = img.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
            canvas.paste(resized_img, (x0, y0))

    return _factory


@pytest.fixture
def draw_wood():
    def _factory(canvas: Image.Image, bbox: tuple[int, int, int, int]):
        x0, y0, x1, y1 = bbox
        with Image.open(Path("examples/EX-EX_0001.00-002.00.jpg")) as img:
            img = img.crop((300, 2100, 5000, 2200))
            resized_img = img.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
            canvas.paste(resized_img, (x0, y0))

    return _factory


@pytest.fixture
def make_metadata(tmp_path):
    def _factory(
        depth_start: float,
        depth_end: float,
        draw_fn: Callable[[Image.Image], None] = lambda draw: None,
        size: tuple[int, int] = _IMG_SIZE,
        is_override: bool = False,
    ) -> ImageMetadataCores:
        """Creates an ImageMetadataCores pointing to a synthetic TIF image built by draw_fn.

        Args:
            depth_start (float): Top-of-core depth in metres, encoded into the filename.
            depth_end (float): Bottom-of-core depth in metres, encoded into the filename.
            draw_fn (Callable[[Image.Image], None]): Callback that draws onto the
                background image (e.g. to add a core or tray rectangle). Defaults to a no-op,
                producing a flat background image.
            size (tuple[int, int]): Size of the synthetic image in pixels. Defaults to _IMG_SIZE.
            is_override (bool): Whether this represents a manually cropped override image.

        Returns:
            ImageMetadataCores: Metadata pointing at the saved synthetic TIF image.
        """
        filename = f"{uuid.uuid4()}_{depth_start:07.2f}-{depth_end:07.2f}_vd_p.TIF"
        image_path = tmp_path / filename
        img = Image.new("RGB", size, color=_BACKGROUND_COLOR)
        draw_fn(img)
        img.save(image_path)
        return ImageMetadataCores(
            borehole_id="GBC-CB50",
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
            is_override=is_override,
        )

    return _factory
