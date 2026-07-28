"""Ruler detection: per-image OCR and aggregation across a shape group."""

import numpy as np
import pytesseract
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from sklearn.metrics import pairwise_distances

from src.config import SegmentationRulerConfig
from src.models import ImageMetadata, RulerSegmentResult
from src.segment.utils.misc import ProcessGroupByShape


def segment_ruler(img_metadata: ImageMetadata, config: SegmentationRulerConfig) -> RulerSegmentResult | None:
    """Detect a depth ruler by OCR'ing its printed number ticks and derive a pixel-to-unit scale.

    Binarizes the image for more reliable OCR, keeps digit detections within
    [text_min_value, text_max_value], and drops detections whose spacing deviates
    from the median step between consecutive numbers (outliers).

    Args:
        img_metadata (ImageMetadata): Metadata of the image to load and segment.
        config (SegmentationRulerConfig): Tunable segmentation parameters.

    Returns:
        RulerSegmentResult | None: Bounding box enclosing all detected ruler numbers, the
            pixel-per-unit scale, and the per-number bounding boxes, or None if no ruler
            numbers were detected.
    """
    img = img_metadata.load_image(factor=config.downscale_factor)

    # OCR performs better on binarized images
    img_gray = rgb2gray(img)
    local_thresh = threshold_otsu(img_gray)
    img_bin = (img_gray > local_thresh).astype(np.uint8)

    # Run OCR
    img_data = pytesseract.image_to_data(255 * img_bin, output_type=pytesseract.Output.DICT)

    data = np.array(
        # Only keep text from text_min_value to text_max_value
        [
            (int(text), left, top, width, height)
            for text, left, top, width, height in zip(
                img_data["text"], img_data["left"], img_data["top"], img_data["width"], img_data["height"], strict=True
            )
            if text.isdigit() and config.text_min_value <= int(text) <= config.text_max_value
        ]
    )

    if data.size == 0:
        return None

    # Central point of detected number (left + width/2, top + height / 2)
    X = data[:, [1, 2]] + data[:, [3, 4]] / 2
    y = data[:, 0]

    # Sort values in increasing order and compute steps / median step (robust to outliers)
    y_sort = np.argsort(y)
    X_diff = np.linalg.norm(np.diff(X[y_sort], axis=0), axis=1)
    y_diff = np.diff(y[y_sort], axis=0)
    steps_median = np.median(X_diff[y_diff != 0] / y_diff[y_diff != 0]).item()

    # Drop detections that are not aligned with detected steps (distance to neighbor)
    distances = pairwise_distances(X) / (pairwise_distances(y[:, None]) + 1e-16)
    distances_idx = ~np.eye(distances.shape[0], dtype=bool)
    distances = distances[distances_idx].reshape(
        # Remove diagonal and reshape NxN -> Nx(N-1)
        (
            distances.shape[0],
            distances.shape[1] - 1,
        )
    )
    id_inliers = abs(np.median(distances, axis=1) - steps_median) / steps_median < config.r_error_outliers

    if not id_inliers.any():
        return None

    # Reconstruct bbox for each unit (left, top, left + width, top + height)
    bbox_units = np.concatenate(
        (
            data[id_inliers][:, [1, 2]],
            data[id_inliers][:, [1, 2]] + data[id_inliers][:, [3, 4]],
        ),
        axis=1,
    )
    bbox_units = (1 / config.downscale_factor) * bbox_units

    return RulerSegmentResult(
        bbox=(
            bbox_units[:, 0].min().item(),
            bbox_units[:, 1].min().item(),
            bbox_units[:, 2].max().item(),
            bbox_units[:, 3].max().item(),
        ),
        px_per_unit=(1 / config.downscale_factor) * steps_median,
        bbox_units=[tuple(row) for row in bbox_units.tolist()],
    )


class ProcessRulerGroupByShape(ProcessGroupByShape[RulerSegmentResult, RulerSegmentResult]):
    """Detect a shared depth-ruler scale for a group of same-shaped images."""

    def __init__(
        self,
        config: SegmentationRulerConfig,
    ):
        """Configure the ruler group segmentation.

        Args:
            config (SegmentationRulerConfig): Tunable segmentation parameters.
        """
        super().__init__(min_group_size=config.n_min_ruler, n_workers=config.n_workers, seed=config.seed)
        self.config = config

    def _preprocess(self, img_metadata: ImageMetadata) -> RulerSegmentResult | None:
        """Run ruler OCR detection on a single image.

        Args:
            img_metadata (ImageMetadata): Metadata of the image to load and segment.

        Returns:
            RulerSegmentResult | None: Detected ruler result, or None.
        """
        return segment_ruler(img_metadata, self.config)

    def _aggregate(self, processed_items: list[RulerSegmentResult]) -> RulerSegmentResult | None:
        """Pick the median-scale ruler detection from the group.

        Args:
            processed_items (list[RulerSegmentResult]): Per-image ruler detections for one shape
                group. OCR hit-rate is low so processed_items might contain fewer samples than expected.

        Returns:
            RulerSegmentResult | None: The detection whose `px_per_unit` is the median across
                the group, or None if no detections are available.
        """
        if not processed_items:
            return None

        return sorted(processed_items, key=lambda d: d.px_per_unit)[len(processed_items) // 2]
