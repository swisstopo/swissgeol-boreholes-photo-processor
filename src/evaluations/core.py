"""Module for checking core width and length consistency across a set of detections."""

from abc import ABC, abstractmethod

import numpy as np

from src.evaluations.config import (
    CoreCheckConfig,
    CoreCheckResult,
    CoreLengthCheckConfig,
    CoreValueCheckResult,
    EvaluationConfig,
)
from src.models import ImageMetadataProcessed


class EvaluationCompute(ABC):
    """Base class for flagging cores whose measured value deviates too far from a group median.

    Subclasses implement `_compute` to derive the per-detection measured value. This base class
    handles computing the group median reference from those values and flagging entries whose
    relative deviation from it exceeds `relative_tolerance`.

    Args:
        config (CoreCheckConfig): Tunable parameters (relative_tolerance, min_samples) for the check.
    """

    def __init__(self, config: CoreCheckConfig):
        self.min_samples = config.min_samples
        self.relative_tolerance = config.relative_tolerance

    def _evaluate_median(
        self,
        values: list[float | None],
    ) -> list[CoreValueCheckResult | None]:
        """Flag values that deviate too far from the group median of all measured values.

        None entries mark a detection with no measurement for this check (e.g. a missing core or
        ruler detection).

        Args:
            values (list[float | None]): Measured value for each detection, or None where unavailable.

        Returns:
            list[CoreCheckOutcome | None]: One entry per detection. None when fewer than `min_samples` values.
        """
        values_ = np.array(values, dtype=float)

        if np.isfinite(values_).sum() < self.min_samples:
            return [None] * len(values)

        reference = float(np.nanmedian(values_))
        relative_errors = np.abs(values_ - reference) / reference

        return [
            CoreValueCheckResult(
                bool(relative_error <= self.relative_tolerance), float(relative_error), value, reference
            )
            if value is not None
            else None
            for relative_error, value in zip(relative_errors, values, strict=True)
        ]

    def evaluate(
        self,
        detections: list[ImageMetadataProcessed],
    ) -> list[CoreValueCheckResult | None]:
        """Compute each detection's measured value via `_compute` and flag deviations from the median.

        Args:
            detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.

        Returns:
            list[CoreCheckOutcome | None]: One entry per detection. None where the check was skipped.
        """
        return self._evaluate_median(values=[self._compute(detection) for detection in detections])

    @abstractmethod
    def _compute(self, detection: ImageMetadataProcessed) -> float | None:
        """Derive the measured value for one detection.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: The measured value, or None if it can't be computed.
        """
        raise NotImplementedError()


class EvaluationWidthCompute(EvaluationCompute):
    """Flags cores whose width deviates too far from the median width."""

    def _compute(self, detection: ImageMetadataProcessed) -> float | None:
        """Compute a core's width in pixels: its bounding box's vertical (y) extent.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: Width in pixels, or None if no core was detected for this image.
        """
        return detection.core.bbox[3] - detection.core.bbox[1] if detection.core is not None else None


class EvaluationLengthCompute(EvaluationCompute):
    """Flags cores whose length-to-depth ratio deviates too far from the median ratio.

    Args:
        config (CoreLengthCheckConfig): Tunable parameters (relative_tolerance, min_samples,
            max_depth_range) for the core length check.
    """

    def __init__(self, config: CoreLengthCheckConfig):
        super().__init__(config)
        self.max_depth_range = config.max_depth_range

    def _compute(self, detection: ImageMetadataProcessed) -> float | None:
        """Compute a core's length-to-depth ratio, normalized by the ruler's px-per-unit scale.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: The normalized px-per-depth-unit ratio, or None.
        """
        if detection.core is None or detection.ruler is None:
            return None

        depth_range = min(detection.depth_end - detection.depth_start, self.max_depth_range)
        ruler_res = detection.ruler.px_per_unit

        if not depth_range or not ruler_res:
            return None

        return (detection.core.bbox[2] - detection.core.bbox[0]) / depth_range / detection.ruler.px_per_unit


def evaluate_detections(detections: list[ImageMetadataProcessed], config: EvaluationConfig) -> list[CoreCheckResult]:
    """Run the width and length checks and merge them into one result per file.

    Args:
        detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.
        config (EvaluationConfig): Configuration parameters for all core checks.

    Returns:
        list[CoreCheckResult]: One result per detection. width/length are None for a file
        whose corresponding check was skipped (see CoreCheckConfig.min_samples).
    """
    width_results = EvaluationWidthCompute(config=config.core_width).evaluate(detections)
    length_results = EvaluationLengthCompute(config=config.core_length).evaluate(detections)

    return [
        CoreCheckResult(filename=detection.image_path.name, width=width, length=length)
        for detection, width, length in zip(detections, width_results, length_results, strict=True)
    ]
