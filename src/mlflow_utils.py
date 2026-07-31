"""Utility functions for MLflow."""

import csv
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from PIL import Image, ImageDraw, ImageFont

from src.evaluations.config import CoreCheckResult
from src.models import ImageMetadataProcessed, ImageSegmentResult, TraySegmentResult
from src.utils import scale_bbox


def log_tray_segment_mlflow(
    result: TraySegmentResult | None,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
) -> None:
    """Log the debug background/foreground images used to estimate a shared tray bbox to MLflow.

    Draws the estimated bbox on the foreground (per-pixel std) image for visual inspection, and
    logs both the mean background image and the annotated foreground image as separate artifacts.

    Args:
        result (TraySegmentResult | None): If its img_background/img_foreground debug images
            are unset, the corresponding artifact is skipped.
        filename (str): The filename prefix for the artifacts.
        suffix (str, optional): File extension (including the dot) used when saving the artifacts.
            Defaults to ".jpg".
        subfolder (str | None, optional): Optional subfolder for image logging. Defaults to None.
    """
    if result is None:
        return

    if result.img_background is not None:
        img_bg_pil = Image.fromarray((result.img_background * 255).astype(np.uint8))
        log_artifact_with_mlflow(img_bg_pil, filename + "_background", suffix, subfolder)

    if result.img_foreground is not None and result.img_downscale_factor is not None:
        img_fg_pil = Image.fromarray(
            (result.img_foreground / (result.img_foreground.max() + 1e-16) * 255).astype(np.uint8)
        ).convert("RGB")
        draw = ImageDraw.Draw(img_fg_pil)
        draw.rectangle(scale_bbox(result.bbox, result.img_downscale_factor), outline="red", width=5)
        log_artifact_with_mlflow(img_fg_pil, filename + "_foreground", suffix, subfolder)


def upload_log_to_mlflow(log_path: Path) -> None:
    """Flush all logging handlers and upload the log file to MLflow as an artifact.

    Args:
        log_path (Path): Path to the log file to upload.
    """
    for handler in logging.root.handlers:
        handler.flush()
    mlflow.log_artifact(str(log_path))


def log_image_metadata_processed_mlflow(
    result: ImageMetadataProcessed,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
    font_size: int = 30,
    run_id: str | None = None,
) -> None:
    """Log a processed image to MLflow with core/tray/ruler bounding boxes overlaid.

    Args:
        result (ImageMetadataProcessed): The processed image whose detected regions are drawn and logged.
        filename (str): The filename prefix for the artifact.
        suffix (str): File extension (including the dot) used when saving the artifact, e.g. ".jpg" or ".png".
        subfolder (str | None): Optional subfolder for image logging.
        font_size (int): Font size used to draw the ruler's px-per-unit label.
        run_id (str | None): If set, log the artifact directly to this run via MlflowClient
            instead of the active run. Safe to call concurrently from multiple processes
            against the same run_id, unlike `mlflow.start_run`/`end_run`. Defaults to None.
    """
    img_npy = result.load_image()
    img_pil = Image.fromarray((img_npy * 255).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=font_size)

    if result.core is not None:
        # Draw segments < overall detection
        for bbox in result.core.bbox_segments or []:
            draw.rectangle(bbox, outline=(0, 255, 0, 255), fill=(0, 255, 0, 30), width=2)
        draw.rectangle(result.core.bbox, outline=(0, 255, 0, 255), width=5)

    if result.ruler is not None:
        # Draw units < overall detection < resolution text
        for bbox in result.ruler.bbox_units or []:
            draw.rectangle(bbox, outline=(0, 0, 255, 255), fill=(0, 0, 255, 30), width=2)
        draw.rectangle(result.ruler.bbox, outline=(0, 0, 255, 255), width=5)
        draw.text(
            (result.ruler.bbox[0], result.ruler.bbox[1]),
            f"{result.ruler.px_per_unit:.1f} px/unit",
            fill=(255, 255, 255, 255),
            font=font,
            anchor="lt",
        )

    if result.tray is not None:
        draw.rectangle(result.tray.bbox, outline=(255, 0, 0, 255), width=5)

    img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")

    log_artifact_with_mlflow(img_pil, filename, suffix, subfolder, run_id=run_id)


def log_segmentation_results_with_mlflow(
    detections: list[ImageMetadataProcessed],
    time: float,
) -> None:
    """Log a summary of the segmentation timing and approach breakdown to MLflow.

    Dumps a single JSON artifact ("segmentation_summary.json") containing the overall
    time for the run, a per-approach (single vs. shared group) count/timing breakdown
    for the tray, ruler, and core detectors, and every image's full detection result.

    Args:
        detections (list[ImageMetadataProcessed]): Per-image processed results.
        time (float): Overall wall-clock time, in seconds, for the segmentation run.
    """
    mlflow.log_dict(
        {
            "time": {
                "overall": time,
                "tray": ImageSegmentResult.approach_to_json([detection.tray for detection in detections]),
                "ruler": ImageSegmentResult.approach_to_json([detection.ruler for detection in detections]),
                "core": ImageSegmentResult.approach_to_json([detection.core for detection in detections]),
            },
            "detections": [detection.to_dict() for detection in detections],
        },
        "segmentation_summary.json",
    )


def log_artifact_with_mlflow(
    img: Image.Image,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
    run_id: str | None = None,
) -> None:
    """Log an image artifact to MLflow.

    Args:
        img (Image.Image): The image to log.
        filename (str): The filename prefix for the artifact.
        suffix (str): File extension (including the dot) used when saving the artifact, e.g. ".jpg" or ".png".
        subfolder (str | None): Optional subfolder for image logging.
        run_id (str | None): If set, log the artifact directly to this run via MlflowClient
            instead of the active run. Safe to call concurrently from multiple processes
            against the same run_id, unlike `mlflow.start_run`/`end_run`. Defaults to None.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / f"{filename}{suffix}"
        img.save(artifact_path)
        if run_id is not None:
            MlflowClient().log_artifact(run_id=run_id, local_path=str(artifact_path), artifact_path=subfolder)
        else:
            mlflow.log_artifact(local_path=str(artifact_path), artifact_path=subfolder)


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
