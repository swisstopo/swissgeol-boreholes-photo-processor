"""Utility functions for MLflow."""

import tempfile
from pathlib import Path

import mlflow
from PIL import Image, ImageDraw


def log_artifact_with_mlflow(
    img: Image.Image,
    filename: str,
    bounding_box: tuple[float, float, float, float] | None = None,
    suffix: str = ".jpg",
    subfolder: str | None = None,
) -> None:
    """Log an artifact to MLflow.

    Args:
        img (Image.Image): The image to log.
        filename (str): The filename for the artifact.
        bounding_box (tuple[float, float, float, float] | None): The bounding box coordinates, if applicable.
        suffix (str): File extension (including the dot) used when saving the artifact, e.g. ".jpg" or ".png".
        subfolder (str | None): Optional subfolder for image logging.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / f"{filename}{suffix}"

        if bounding_box is not None:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            draw.rectangle(bounding_box, outline="red", width=2)

        img.save(artifact_path)
        mlflow.log_artifact(
            local_path=str(artifact_path),
            artifact_path=subfolder,
        )
