"""Utility functions for MLflow."""

import tempfile
from dataclasses import fields
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


def log_evaluation_results_with_mlflow(
    results: list,
    filename: str,
) -> None:
    """Log evaluation results to MLflow.

    Shared across all CoreCheckResults subtypes (width, length, ...): the pass-rate
    score and the failed-core dump are common to every check. Folder-level reference
    fields (named `folder_*`, e.g. folder_median_width) are the same on every result,
    so they're additionally logged as their own metric.

    Args:
        results (list): The evaluation results to log (instances of a CoreCheckResults subclass).
        filename (str): The filename for the artifact.
    """
    if not results:
        return

    score = sum(r.passed for r in results) / len(results)
    mlflow.log_metric(f"{filename}_score", score)

    folder_field_names = {f.name for f in fields(results[0]) if f.name.startswith("folder_")}
    for folder_field_name in folder_field_names:
        mlflow.log_metric(f"{filename}_{folder_field_name}", getattr(results[0], folder_field_name))

    per_core_field_names = {f.name for f in fields(results[0])} - folder_field_names - {"filename", "passed"}
    failed_cores = {
        r.filename: {name: getattr(r, name) for name in per_core_field_names} for r in results if not r.passed
    }
    if failed_cores:
        mlflow.log_dict(failed_cores, f"failed_{filename}.json")
