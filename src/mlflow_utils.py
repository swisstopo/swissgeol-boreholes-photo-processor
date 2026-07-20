"""Utility functions for MLflow."""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import ImageMetadataProcessed


def log_image_metadata_processed_mlflow(
    result: ImageMetadataProcessed,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
    font_size: int = 30,
) -> None:
    """Log a processed image to MLflow with core/tray/ruler bounding boxes overlaid.

    Args:
        result (ImageMetadataProcessed): The processed image whose detected regions are drawn and logged.
        filename (str): The filename prefix for the artifact.
        suffix (str): File extension (including the dot) used when saving the artifact, e.g. ".jpg" or ".png".
        subfolder (str | None): Optional subfolder for image logging.
        font_size (int): Font size used to draw the ruler's px-per-unit label.
    """
    img_npy = result.load_image()
    img_pil = Image.fromarray((img_npy * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default(size=font_size)

    if result.core:
        draw.rectangle(result.core.bbox, outline="green", width=5)
    if result.ruler:
        draw.rectangle(result.ruler.bbox, outline="blue", width=5)
        for bbox in result.ruler.bbox_units:
            draw.rectangle(bbox, outline="blue", width=2)
        draw.text(
            (result.ruler.bbox[0], result.ruler.bbox[1]),
            f"{result.ruler.px_per_unit:.1f} px/unit",
            fill=(255, 255, 255),
            font=font,
            anchor="lt",
        )

    if result.tray:
        draw.rectangle(result.tray.bbox, outline="red", width=5)

    log_artifact_with_mlflow(img_pil, filename, suffix, subfolder)


def log_artifact_with_mlflow(
    img: Image.Image,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
) -> None:
    """Log an image artifact to MLflow.

    Args:
        img (Image.Image): The image to log.
        filename (str): The filename prefix for the artifact.
        suffix (str): File extension (including the dot) used when saving the artifact, e.g. ".jpg" or ".png".
        subfolder (str | None): Optional subfolder for image logging.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / f"{filename}{suffix}"
        img.save(artifact_path)
        mlflow.log_artifact(
            local_path=str(artifact_path),
            artifact_path=subfolder,
        )
