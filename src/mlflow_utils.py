"""Utility functions for MLflow."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
from PIL import Image, ImageDraw

from src.config import CoreCheckResult


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


def log_evaluation_results_with_mlflow(
    results: list[CoreCheckResult],
    folder_name: str,
) -> None:
    """Log evaluation results to MLflow.

    Logs the width and length pass-rate as separate metrics, and dumps every file's full
    width/length results as a single JSON artifact, keyed by filename -- useful for inspecting
    a specific core's width and length results side by side, not just the ones that got flagged.
    The artifact is named after the folder so batch runs don't clobber each other's results.

    Args:
        results (list[CoreCheckResult]): Per-file merged core check results.
        folder_name (str): Name of the input folder these results belong to, used as the
            JSON artifact's filename.
    """
    if not results:
        return

    width_results = [r.width for r in results if r.width is not None]
    length_results = [r.length for r in results if r.length is not None]

    if width_results:
        mlflow.log_metric("width_score", sum(r.passed for r in width_results) / len(width_results))
    if length_results:
        mlflow.log_metric("length_score", sum(r.passed for r in length_results) / len(length_results))

    predictions = {
        r.filename: {
            "width": asdict(r.width) if r.width is not None else None,
            "length": asdict(r.length) if r.length is not None else None,
        }
        for r in results
    }
    mlflow.log_dict(predictions, f"{folder_name}.json")
