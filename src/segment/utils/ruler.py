"""Ruler detection: per-image OCR and aggregation across a shape group."""

from timeit import default_timer as timer

import numpy as np
import pytesseract
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from sklearn.metrics import pairwise_distances

from src.config import SegmentationRulerConfig
from src.models import ImageMetadataCores, RulerSegmentResult
from src.segment.utils.misc import ProcessGroupByShape


def _select_inlier_detections(X: np.ndarray, y: np.ndarray, r_error_outliers: float) -> tuple[np.ndarray, float]:
    """Flag detections whose neighbor-consistency count matches the group's consensus.

    Computes the pairwise pixel-distance-per-unit-step between every pair of detections, then
    counts, for each detection, how many others it is step-consistent with (within
    `r_error_outliers`). A detection is an inlier if that neighbor count itself is close to the
    median neighbor count across all detections -- misdetections tend to be consistent with few
    or no other points, even if their own spacing looks locally plausible.

    Args:
        X (np.ndarray): (N, 2) array of detection center coordinates (pixels).
        y (np.ndarray): (N,) array of detection values (e.g. ruler numbers).
        r_error_outliers (float): Relative tolerance for step-consistency.

    Returns:
        tuple[np.ndarray, float]: (N,) boolean inlier mask, and the median step (pixels per unit).
    """
    y_sort = np.argsort(y)
    X_diff = np.linalg.norm(np.diff(X[y_sort], axis=0), axis=1)
    y_diff = np.diff(y[y_sort], axis=0)

    # Sort values in increasing order and compute steps / median step (robust to outliers)
    if not np.any(y_diff != 0):
        # No two detections have distinct values: no reliable scale can be derived.
        return np.zeros(len(y), dtype=bool), 0

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
    # Count number of valid neighbors detected for every entry
    valid_neigh = np.sum(abs(distances - steps_median) / steps_median < r_error_outliers, axis=1)
    # Inliers should be consistent with all neighbors
    median_valid_neigh = np.median(valid_neigh)
    id_inliers = abs(valid_neigh - median_valid_neigh) / (median_valid_neigh + 1e-16) < r_error_outliers

    return id_inliers, steps_median


def segment_ruler(img_metadata: ImageMetadataCores, config: SegmentationRulerConfig) -> RulerSegmentResult | None:
    """Detect a depth ruler by OCR'ing its printed number ticks and derive a pixel-to-unit scale.

    Binarizes the image for more reliable OCR, keeps digit detections within
    [text_min_value, text_max_value], and drops detections whose count of step-consistent
    neighbors deviates from the median neighbor count across all detections (outliers whose
    spacing agrees with few other detections are dropped, even if individually plausible).

    Args:
        img_metadata (ImageMetadataCores): Metadata of the image to load and segment.
        config (SegmentationRulerConfig): Tunable segmentation parameters.

    Returns:
        RulerSegmentResult | None: Bounding box enclosing all detected ruler numbers, the
            pixel-per-unit scale, and the per-number bounding boxes, or None if no ruler
            numbers were detected.
    """
    t_start = timer()
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

    # At least two samples to create interval
    if data.size == 0 or data.shape[0] < 2:
        return None

    # Central point of detected number (left + width/2, top + height / 2)
    X = data[:, [1, 2]] + data[:, [3, 4]] / 2
    y = data[:, 0]

    id_inliers, steps_median = _select_inlier_detections(X, y, config.r_error_outliers)

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
        time=timer() - t_start,
    )


class ProcessRulerGroupByShape(ProcessGroupByShape[RulerSegmentResult, RulerSegmentResult]):
    """Detect a shared depth-ruler scale for a group of same-shaped images."""

    def __init__(
        self,
        config: SegmentationRulerConfig,
        n_workers: int = 1,
    ):
        """Configure the ruler group segmentation.

        Args:
            config (SegmentationRulerConfig): Tunable segmentation parameters.
            n_workers (int): Number of worker processes used to run `_preprocess` in parallel.
        """
        super().__init__(min_group_size=config.n_min_ruler, seed=config.seed, n_workers=n_workers)
        self.config = config

    def _preprocess(
        self, img_metadata: ImageMetadataCores, img_metadata_ref: ImageMetadataCores
    ) -> RulerSegmentResult | None:
        """Run ruler OCR detection on a single image.

        Args:
            img_metadata (ImageMetadataCores): Metadata of the image to load and segment.
            img_metadata_ref (ImageMetadataCores): Reference image.

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
