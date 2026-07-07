"""Utility functions for MLflow."""

import tempfile
from pathlib import Path

import mlflow
from PIL import Image, ImageDraw


def log_artifact_with_mlflow(
    img: Image.Image,
    filename: str,
    bounding_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Log an artifact to MLflow.

    Args:
        img (Image.Image): The image to log.
        filename (str): The filename for the artifact.
        bounding_box (tuple[float, float, float, float] | None): The bounding box coordinates, if applicable.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / f"{filename}.png"

        if bounding_box is not None:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            draw.rectangle(bounding_box, outline="red", width=2)

        img.save(artifact_path)
        mlflow.log_artifact(str(artifact_path))


def log_evaluation_results_with_mlflow(
    results: list,
    filename: str,
) -> None:
    """Log evaluation results to MLflow.

    Args:
        results (list): The evaluation results to log.
        filename (str): The filename for the artifact.
    """
    if results:
        score = sum(r.passed for r in results) / len(results)
        mlflow.log_metric(f"{filename}_score", score)

        failed_cores = {r.filename: r.deviation for r in results if not r.passed}
        if failed_cores:
            mlflow.log_dict(failed_cores, f"{filename}_failed_cores.json")
