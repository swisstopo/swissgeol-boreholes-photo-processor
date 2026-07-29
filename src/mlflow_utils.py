"""Utility functions for MLflow."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
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

    log_artifact_with_mlflow(img_pil, filename, suffix, subfolder)


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
                "tray": ImageSegmentResult.apporach_to_json([detection.tray for detection in detections]),
                "ruler": ImageSegmentResult.apporach_to_json([detection.ruler for detection in detections]),
                "core": ImageSegmentResult.apporach_to_json([detection.core for detection in detections]),
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


def log_evaluation_results_with_mlflow(
    results: list[CoreCheckResult],
) -> None:
    """Log evaluation results to MLflow.

    Logs the width and length pass-rate and mean squared error as separate metrics, and
    dumps every file's full width/length results as a single JSON artifact.

    Args:
        results (list[CoreCheckResult]): Per-file merged core check results.
    """
    if not results:
        return

    checks_by_name = {
        "width": [r.width for r in results if r.width is not None],
        "length": [r.length for r in results if r.length is not None],
    }
    for name, checks in checks_by_name.items():
        if checks:
            mlflow.log_metric(f"{name}_acc", sum(c.passed for c in checks) / len(checks))
            mlflow.log_metric(f"{name}_mre", sum(c.relative_error for c in checks) / len(checks))

    predictions = {
        r.filename: {
            "width": asdict(r.width) if r.width is not None else None,
            "length": asdict(r.length) if r.length is not None else None,
        }
        for r in results
    }
    mlflow.log_dict(predictions, "evaluation.json")
