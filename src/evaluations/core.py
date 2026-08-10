"""Module for checking core width and length consistency across a set of detections."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.evaluations.config import (
    CoreCheckConfig,
    CoreCheckResult,
    CoreLengthCheckConfig,
    CoreValueCheckResult,
    CoreWidthCheckConfig,
    EvaluationConfig,
)
from src.models import ImageMetadataProcessed


class DPCoreWidthEstimation:
    """Splits an ordered sequence of measurements into segments.

    Uses dynamic programming to find, for each candidate segment count K, the partition into K
    segments that minimizes the total squared error to each segment's mean. Increases K from 1 up
    to `max_k` and stops as soon as adding a segment no longer reduces the error by more than `alpha`.
    """

    def __init__(self, max_k: int = 2, alpha: float = 0.25):
        """Initialize the estimator.

        Args:
            max_k (int, optional): Maximum number of segments to try. Defaults to 2.
            alpha (float, optional): Minimum relative error improvement required to accept an
                additional segment. Defaults to 0.25.
        """
        self.max_k = max_k
        self.alpha = alpha
        self._segments = []
        self._err = np.inf

    def fit(self, y: np.ndarray) -> None:
        """Fit the best segmentation of `y` using DP.

        Args:
            y (np.ndarray): 1D array of values to segment, ordered by index.
        """
        self._err, self._segments = self._forward(y, K=1)

        for k in range(2, self.max_k):
            err, segments = self._forward(y, K=k)
            if abs(err - self._err) / self._err < self.alpha:
                break

            self._segments = segments
            self._err = err

    def _forward(self, y: np.ndarray, K: int) -> tuple[float, list[tuple[int, int]]]:
        """Fit y with exactly K horizontal segments, minimizing total squared error.

        Args:
            y (np.ndarray): 1D array of values to segment, ordered by index.
            K (int): Exact number of segments to fit.

        Returns:
            tuple[float, list[tuple[int, int]]]: The total squared error of the best fit, and the
                resulting segments as (start, end) index pairs from the DP backtracking.

        Example:
            Let's assume input sample where we want k=3 steps
            y = [1.0, 1.2, 0.9, 5.0, 5.3, 9.0, 9.4]

            Estimated cost (inf is unreachable)
            dp     i=0      1       2       3       4       5       6       7
            k=0  0.000      inf     inf     inf     inf     inf     inf     inf
            k=1    inf      0.000   0.020   0.047   11.848  20.428  53.713  81.237
            k=2    inf      inf     0.00    0.020   0.047   0.092   9.973   16.574
            k=3    inf      inf     inf     0.000   0.020   0.047   0.092   0.173

            # Store segment (breakponts) decision
            brk    i=0      1       2       3       4       5       6       7
            k=0      -      -       -       -       -       -       -       -
            k=1      -      0       0       0*      0       0       0       0
            k=2      -      -       1       2       3       3*      3       3
            k=3      -      -       -       2       3       4       5       5*

            # Backtracking
            (k=3, i=7)  brk=5  →  segment 6..7   →  go to (k=2, i=5)
            (k=2, i=5)  brk=3  →  segment 4..5   →  go to (k=1, i=3)
            (k=1, i=3)  brk=0  →  segment 1..3   →  go to (k=0, i=0)
        """
        n = len(y)

        def cost(a: int, b: int) -> float:
            """Sum of squared deviations from the mean for segment."""
            m = b - a + 1
            mean = np.sum(y[a - 1 : b]) / m
            return np.sum([(v - mean) ** 2 for v in y[a - 1 : b]])

        dp = np.ones((K + 1, n + 1)) * float("inf")
        brk = np.zeros((K + 1, n + 1), dtype=int)

        dp[0][0] = 0.0
        for k in range(1, K + 1):
            for i in range(1, n + 1):
                for j in range(k - 1, i):
                    c = dp[k - 1][j] + cost(j + 1, i)  # <-- the O(n) hidden here
                    if c < dp[k][i]:
                        dp[k][i] = c
                        brk[k][i] = j

        segments, i = [], n
        for k in range(K, 0, -1):
            j = brk[k][i].item()
            segments.append((j, i))
            i = j

        return dp[K][n].item(), segments[::-1]


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
        self, values: Sequence[float | None], segments: list[tuple[int, int]]
    ) -> list[CoreValueCheckResult | None]:
        """Measure values that deviate too far from the median of their own segment.

        Args:
            values (Sequence[float | None]): Measured value for each detection, or None where unavailable.
            segments (list[tuple[int, int]]): Contiguous (start, end) index ranges into `values`;
                the median reference is computed independently within each segment.

        Returns:
            list[CoreValueCheckResult | None]: One result per value, concatenated in the given
                segment order. None for a segment with fewer than `min_samples` finite values.
        """
        values_ = np.array(values, dtype=float)

        results: list[CoreValueCheckResult | None] = []
        for start, end in segments:
            values_segment = values_[start:end]

            if np.isfinite(values_segment).sum() < self.min_samples:
                results.extend([None] * len(values_segment))
            else:
                reference_segment = float(np.nanmedian(values_segment))
                relative_errors = np.abs(values_segment - reference_segment) / (reference_segment + 1e-16)
                results.extend(
                    [
                        CoreValueCheckResult(
                            bool(relative_error <= self.relative_tolerance),
                            float(relative_error),
                            value,
                            reference_segment,
                        )
                        if value is not None
                        else None
                        for relative_error, value in zip(relative_errors, values_segment, strict=True)
                    ]
                )
        return results

    def evaluate(
        self,
        detections: list[ImageMetadataProcessed],
    ) -> list[CoreValueCheckResult | None]:
        """Compute each detection's measured value and results from the median.

        Args:
            detections (list[ImageMetadataProcessed]): List of processed image metadata with detection results.

        Returns:
            list[CoreValueCheckResult | None]: One entry per detection. None where the check was skipped.
        """
        values = [self._mesure(detection) for detection in detections]
        values_filtered = [value for value in values if value is not None]

        segments_filtered = self._estimate_segments(values_filtered)
        results_filtered = self._evaluate_median(values=values_filtered, segments=segments_filtered)

        # Put None back if not able to predict result
        it = iter(results_filtered)
        return [None if value is None else next(it) for value in values]

    def _estimate_segments(self, values: list[float]) -> list[tuple[int, int]]:
        """Return the whole sequence as a single segment.

        Subclasses may override this to split `values` into multiple contiguous segments (e.g. to
        fit several reference groups instead of one global median).

        Args:
            values (list[float]): Measured values to segment, in detection order.

        Returns:
            list[tuple[int, int]]: Segment boundaries as (start, end) index pairs into `values`.
        """
        return [(0, len(values) - 1)]

    @abstractmethod
    def _mesure(self, detection: ImageMetadataProcessed) -> float | None:
        """Derive the measured value for one detection.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: The measured value, or None if it can't be computed.
        """
        raise NotImplementedError()


class EvaluationWidthCompute(EvaluationCompute):
    """Flags cores whose width deviates too far from the median width.

    Args:
        config (CoreWidthCheckConfig): Tunable parameters for the core width check.
    """

    def __init__(self, config: CoreWidthCheckConfig):
        super().__init__(config)
        self.max_width_steps = config.max_width_steps
        self.relative_tolerance_steps = config.relative_tolerance_steps

    def _mesure(self, detection: ImageMetadataProcessed) -> float | None:
        """Compute a core's width in pixels: its bounding box's vertical (y) extent.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: Width in pixels, or None if no core was detected for this image.
        """
        if detection.core is None or detection.ruler is None or detection.ruler.px_per_unit is None:
            return None

        return (detection.core.bbox[3] - detection.core.bbox[1]) / detection.ruler.px_per_unit

    def _estimate_segments(self, values: list[float]) -> list[tuple[int, int]]:
        """Split width values into segments using dynamic-programming change-point detection.

        Args:
            values (list[float]): Measured core widths, in detection order.

        Returns:
            list[tuple[int, int]]: Segment boundaries as (start, end) index pairs into `values`,
                found by `DPCoreWidthEstimation`.
        """
        estimator = DPCoreWidthEstimation(max_k=self.max_width_steps, alpha=self.relative_tolerance_steps)
        estimator.fit(np.array(values))
        return estimator._segments


class EvaluationLengthCompute(EvaluationCompute):
    """Flags cores whose length-to-depth ratio deviates too far from the median ratio.

    Args:
        config (CoreLengthCheckConfig): Tunable parameters (relative_tolerance, min_samples,
            max_depth_range) for the core length check.
    """

    def __init__(self, config: CoreLengthCheckConfig):
        super().__init__(config)
        self.max_depth_range = config.max_depth_range

    def _mesure(self, detection: ImageMetadataProcessed) -> float | None:
        """Compute a core's length-to-depth ratio, normalized by the ruler's px-per-unit scale.

        Args:
            detection (ImageMetadataProcessed): The processed image metadata to measure.

        Returns:
            float | None: Dimensionless ratio of the core's length-to-depth scale to the ruler's
                detected px-per-unit scale.
        """
        if detection.core is None or detection.ruler is None:
            return None

        depth_range = min(detection.depth_end - detection.depth_start, self.max_depth_range)
        ruler_res = detection.ruler.px_per_unit

        if not depth_range or not ruler_res:
            return None

        return (detection.core.bbox[2] - detection.core.bbox[0]) / depth_range / ruler_res


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
        CoreCheckResult(
            filename=detection.image_path.name,
            width=width,
            length=length,
        )
        for detection, width, length in zip(detections, width_results, length_results, strict=True)
    ]
