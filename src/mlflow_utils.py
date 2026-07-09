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

        if hasattr(results[0], "folder_median_width"):
            mlflow.log_metric(f"{filename}_median", results[0].folder_median_width)
            failed_cores = {r.filename: {"width": r.width, "deviation": r.deviation} for r in results if not r.passed}
            if failed_cores:
                mlflow.log_dict(failed_cores, f"failed_{filename}.json")

        if hasattr(results[0], "folder_ratio_px_per_m"):
            mlflow.log_metric(f"{filename}_folder_ratio_px_per_m", results[0].folder_ratio_px_per_m)
            failed_cores = {
                r.filename: {
                    "length_px": r.length_px,
                    "expected_length_px": r.expected_length_px,
                    "deviation": r.deviation,
                }
                for r in results
                if not r.passed
            }
            if failed_cores:
                mlflow.log_dict(failed_cores, f"failed_{filename}.json")
