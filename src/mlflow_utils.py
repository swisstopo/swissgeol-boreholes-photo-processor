"""Utility functions for MLflow."""

import csv
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.evaluations.config import CoreCheckResult
from src.models import ImageMetadataProcessed, SegmentationRecord


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


def log_segmentation_summary_mlflow(
    num_foreground_groups: int,
    images: list[SegmentationRecord],
    filename: str = "segmentation_summary.json",
) -> None:
    """Log a JSON summary of the segmentation approach used per image.

    Args:
        num_foreground_groups (int): Number of image-shape groups with a successfully
            estimated shared foreground.
        images (list[SegmentationRecord]): Per-image segmentation approach records.
        filename (str): The filename for the JSON artifact.
    """
    mlflow.log_dict(
        {"num_foreground_groups": num_foreground_groups, "images": [asdict(image) for image in images]}, filename
    )


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


def _summarize_checks(results: list[CoreCheckResult]) -> dict[str, tuple[float, float] | None]:
    """Compute each check's pass-rate and mean relative error across a folder's results.

    Args:
        results (list[CoreCheckResult]): Per-file merged core check results for one folder.

    Returns:
        dict[str, tuple[float, float] | None]: Maps "width"/"length" to (pass_rate, mean_relative_error),
            or None when the check was skipped for every file in the folder.
    """
    checks_by_name = {
        "width": [r.width for r in results if r.width is not None],
        "length": [r.length for r in results if r.length is not None],
    }
    return {
        name: (sum(c.passed for c in checks) / len(checks), sum(c.relative_error for c in checks) / len(checks))
        if checks
        else None
        for name, checks in checks_by_name.items()
    }


def log_evaluation_results_with_mlflow(
    results: list[CoreCheckResult],
    folder_name: str,
) -> None:
    """Log evaluation results to MLflow.

    Logs the width and length pass-rate and mean relative error as separate metrics, and
    dumps every file's full width/length results as a single JSON artifact, keyed by filename --
    useful for inspecting a specific core's width and length results side by side, not just the
    ones that got flagged. The artifact is named after the folder so batch runs don't clobber
    each other's results.

    Args:
        results (list[CoreCheckResult]): Per-file merged core check results.
        folder_name (str): Name of the input folder these results belong to, used as the
            JSON artifact's filename.
    """
    if not results:
        return

    for name, summary in _summarize_checks(results).items():
        if summary is not None:
            acc, mre = summary
            mlflow.log_metric(f"{name}_acc", acc)
            mlflow.log_metric(f"{name}_mre", mre)

    predictions = {
        r.filename: {
            "width": asdict(r.width) if r.width is not None else None,
            "length": asdict(r.length) if r.length is not None else None,
        }
        for r in results
    }
    mlflow.log_dict(predictions, f"{folder_name}.json")


def write_evaluation_summary_csv(
    results: list[CoreCheckResult],
    folder_name: str,
    count: int,
    csv_path: Path,
) -> None:
    """Append one folder's width/length pass-rate and mean relative error to a summary CSV.

    Writes the header row if the file doesn't exist yet, otherwise appends. Used to track
    evaluation quality across all folders of a batch run in a single, easy-to-skim file.

    Args:
        results (list[CoreCheckResult]): Per-file merged core check results for one folder.
        folder_name (str): Name of the input folder these results belong to.
        count (int): Number of images processed in this folder.
        csv_path (Path): Path to the summary CSV file to append to.
    """
    if not results:
        return

    summaries = _summarize_checks(results)
    fieldnames = ["folder", "count", "width_acc", "width_mre", "length_acc", "length_mre"]
    row: dict[str, str | int | float] = {"folder": folder_name, "count": count}
    for name in ("width", "length"):
        summary = summaries[name]
        row[f"{name}_acc"] = summary[0] if summary is not None else ""
        row[f"{name}_mre"] = summary[1] if summary is not None else ""

    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
