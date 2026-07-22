"""Utility functions for MLflow."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.evaluations.config import CoreCheckResult
from src.models import ImageMetadataProcessed, TraySegmentResult
from src.utils import scale_bbox


def log_tray_segment_mlflow(
    result: TraySegmentResult | None,
    filename: str,
    suffix: str = ".jpg",
    subfolder: str | None = None,
):
    """Log the debug background/foreground images used to estimate a shared tray bbox to MLflow.

    Draws the estimated bbox on the foreground (per-pixel std) image for visual inspection, and
    logs both the mean background image and the annotated foreground image as separate artifacts.

    Args:
        result (TraySegmentResult): Result of segment_tray_multiple; must include the
            img_background/img_forground debug images.
        filename (str): The filename prefix for the artifacts.
        suffix (str, optional): File extension (including the dot) used when saving the artifacts.
            Defaults to ".jpg".
        subfolder (str | None, optional): Optional subfolder for image logging. Defaults to None.
    """
    if result is None:
        return

    if result.img_background:
        img_bg_pil = Image.fromarray((result.img_background * 255).astype(np.uint8))
        log_artifact_with_mlflow(img_bg_pil, filename + "-background", suffix, subfolder)

    if result.img_forground and result.img_downscale_factor:
        img_fg_pil = Image.fromarray((result.img_forground / result.img_forground.max() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_fg_pil)
        draw.rectangle(scale_bbox(result.bbox, result.img_downscale_factor), outline="red", width=5)
        log_artifact_with_mlflow(img_fg_pil, filename + "-foreground", suffix, subfolder)


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
        draw.rectangle(result.core.bbox, outline="green", width=10)

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


def log_evaluation_results_with_mlflow(
    results: list[CoreCheckResult],
    folder_name: str,
) -> None:
    """Log evaluation results to MLflow.

    Logs the width and length pass-rate and mean squared error as separate metrics, and
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

    checks_by_name = {
        "width": [r.width for r in results if r.width is not None],
        "length": [r.length for r in results if r.length is not None],
    }
    for name, checks in checks_by_name.items():
        if checks:
            mlflow.log_metric(f"{name}_acc", sum(c.passed for c in checks) / len(checks))
            mlflow.log_metric(f"{name}_mse", sum((c.measure - c.reference) ** 2 for c in checks) / len(checks))

    predictions = {
        r.filename: {
            "width": asdict(r.width) if r.width is not None else None,
            "length": asdict(r.length) if r.length is not None else None,
        }
        for r in results
    }
    mlflow.log_dict(predictions, f"{folder_name}.json")
