"""Module for checking core width and length consistency across a set of detections."""

from abc import ABC, abstractmethod

import numpy as np

from src.evaluations.config import (
    CoreCheckConfig,
    CoreCheckResult,
    CoreLengthCheckConfig,
    CoreValueCheckResult,
    CoreWidthCheckConfig,
    EvaluationConfig,
)
from src.models import ImageMetadataProcessedCores


class DPCoreWidthEstimation:
    """Splits an ordered sequence of measurements into segments.

    Uses dynamic programming to find, for each candidate segment count K, the partition into K
    segments that minimizes the total squared error to each segment's mean. Increases K from 1 up
    to `max_k` and stops as soon as either adding a segment no longer reduces the error by more
    than `alpha`, or the resulting segment means stop being non-increasing (segment means are
    expected to shrink with depth).
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
        self._segments_id = []
        self._references = []
        self._err = np.inf

    def fit(self, depths: list[float], widths: list[float | None]) -> None:
        """Fit the best segmentation of `widths` using DP, ordered by depth.

        Args:
            depths (list[float]): Depth value for each measurement; used to order `widths` and to
                express the resulting segment boundaries as depths instead of indices.
            widths (list[float | None]): Measured core widths to segment, matched by index to `depths`.
        """
        # Filter out None values
        pairs = [(x, y) for x, y in zip(depths, widths, strict=True) if y is not None]
        depths_f, widths_f = map(list, zip(*pairs, strict=True)) if pairs else ([], [])

        # Sort by depth
        depth_sort = np.argsort(depths_f)
        depths_f, widths_f = np.array(depths_f)[depth_sort], np.array(widths_f)[depth_sort]

        self._err, self._segments_id, self._references = self._forward(np.array(widths_f), K=1)

        for k in range(2, self.max_k + 1):
            err, segments_id, references = self._forward(np.array(widths_f), K=k)
            if (
                (abs(err - self._err) / (self._err + 1e-16) < self.alpha)  # Improvement should be substantial
                or np.any(np.diff(references) > 0)  # Should be strictly decreasing
            ):
                break
            else:
                self._err, self._segments_id, self._references = err, segments_id, references

        self._segments = [
            (float(depths_f[id_start]), float(depths_f[id_end])) for id_start, id_end in self._segments_id
        ]

    def _forward(self, y: np.ndarray, K: int) -> tuple[float, list[tuple[int, int]], list[float]]:
        """Fit y with exactly K horizontal segments, minimizing total squared error.

        Args:
            y (np.ndarray): 1D array of values to segment, ordered by index.
            K (int): Exact number of segments to fit.

        Returns:
            tuple[float, list[tuple[int, int]], list[float]]: The total squared error of the best
                fit, the resulting segments as (start, end) index pairs from the DP backtracking,
                and each segment's reference value.

        Example:
            Let's assume input sample where we want k=3 steps
            y = [1.0, 1.2, 0.9, 5.0, 5.3, 9.0, 9.4]

            Estimated cost (inf is unreachable)
            dp     i=0      1       2       3       4       5       6       7
            k=0  0.000      inf     inf     inf     inf     inf     inf     inf
            k=1    inf      0.000   0.020   0.047   11.848  20.428  53.713  81.237
            k=2    inf      inf     0.00    0.020   0.047   0.092   9.973   16.574
            k=3    inf      inf     inf     0.000   0.020   0.047   0.092   0.173

            # Store segment (breakpoints) decision
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
            """Sum of squared deviations from the mean for segment.

            Uses the mean (not the median) so the per-segment cost stays a simple closed-form
            squared-error term, cheap to recompute for every candidate split in the DP.
            """
            m = b - a + 1
            mean = np.sum(y[a - 1 : b]) / m
            return np.sum([(v - mean) ** 2 for v in y[a - 1 : b]])

        dp = np.ones((K + 1, n + 1)) * float("inf")
        brk = np.zeros((K + 1, n + 1), dtype=int)

        dp[0][0] = 0.0
        for k in range(1, K + 1):
            for i in range(1, n + 1):
                for j in range(k - 1, i):
                    c = dp[k - 1][j] + cost(j + 1, i)
                    if c < dp[k][i]:
                        dp[k][i] = c
                        brk[k][i] = j

        segments, i = [], n
        for k in range(K, 0, -1):
            j = brk[k][i].item()
            segments.append((j, i - 1))
            i = j

        # Segment boundaries were fit against the mean (see `cost`), but the reported reference
        # value uses the median instead, so a few outlier widths within a segment don't skew it.
        reference = [np.median(y[a : b + 1]) for a, b in segments]
        return dp[K][n].item(), segments[::-1], reference[::-1]


class EvaluationCompute(ABC):
    """Base class for flagging cores whose measured value deviates too far from a group reference.

    Subclasses implement `_measure` to derive the per-detection measured value. This base class
    handles computing the group reference from those values and flagging entries whose
    relative deviation from it exceeds `relative_tolerance`.

    Args:
        config (CoreCheckConfig): Tunable parameters (relative_tolerance, min_samples) for the check.
    """

    def __init__(self, config: CoreCheckConfig):
        self.min_samples = config.min_samples
        self.relative_tolerance = config.relative_tolerance

    def _evaluate_segments(
        self,
        detections: list[ImageMetadataProcessed],
        measures: list[float | None],
        segments_depth: list[tuple[float, float]],
        segments_value: list[float],
    ) -> list[CoreValueCheckResult | None]:
        """Flag detections whose measured value deviates too far from their depth segment's reference.

        Args:
            detections (list[ImageMetadataProcessed]): Processed image metadata to evaluate.
            measures (list[float | None]): Precomputed list measurements to evaluate
            segments_depth (list[tuple[float, float]]): (start, end) depth interval for each segment.
            segments_value (list[float]): Reference value for each segment, matched by index to
                `segments_depth`.

        Returns:
            list[CoreValueCheckResult | None]: One result per detection, in the given order. None
                for a detection whose depth doesn't fall within any segment.
        """
        results: list[CoreValueCheckResult | None] = []
        for i, detection in enumerate(detections):
            valid_segments = [
                segment_start <= detection.depth_start <= segment_end for segment_start, segment_end in segments_depth
            ]
            measure = measures[i]

            if measure is None or not any(valid_segments):
                results.append(None)

            else:
                id_segment = np.argmax(valid_segments)
                segment_start, segment_end = segments_depth[id_segment]
                reference_segment = segments_value[id_segment]
                relative_error = np.abs(measure - reference_segment) / (reference_segment + 1e-16)

                results.append(
                    CoreValueCheckResult(
                        passed=bool(relative_error <= self.relative_tolerance),
                        relative_error=float(relative_error),
                        measure=measure,
                        reference=reference_segment,
                        segment=(segment_start, segment_end),
                    )
                )

        return results

    def evaluate(
        self,
        detections: list[ImageMetadataProcessedCores],
    ) -> list[CoreValueCheckResult | None]:
        """Compute each detection's measured value and results from the reference.

        Args:
            detections (list[ImageMetadataProcessedCores]): List of processed image metadata with detection results.

        Returns:
            list[CoreValueCheckResult | None]: One entry per detection. None for every detection if
                there are fewer than `min_samples` in total; otherwise None for a detection whose
                measured value is missing or whose depth falls outside every segment.
        """
        measures = [self._measure(detection) for detection in detections]

        if sum(m is not None for m in measures) < self.min_samples:
            return [None] * len(detections)

        segments_depth, segments_value = self._estimate_segments(
            depths=[detection.depth_start for detection in detections],
            values=measures,
        )
        return self._evaluate_segments(detections, measures, segments_depth, segments_value)

    def _estimate_segments(
        self, depths: list[float], values: list[float | None]
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Return the whole depth range as a single segment, referenced by its global median.

        Subclasses may override this to split `values` into multiple depth segments (e.g. to fit
        several reference groups instead of one global median).

        Args:
            depths (list[float]): Starting depth for each detection, matched by index to `values`.
            values (list[float | None]): Measured values to segment, in detection order.

        Returns:
            tuple[list[tuple[float, float]], list[float]]: Segment boundaries as (start, end)
                depth pairs, and the reference value for each segment.
        """
        return [(min(depths), max(depths))], [np.median([value for value in values if value is not None]).item()]

    @abstractmethod
    def _measure(self, detection: ImageMetadataProcessedCores) -> float | None:
        """Derive the measured value for one detection.

        Args:
            detection (ImageMetadataProcessedCores): The processed image metadata to measure.

        Returns:
            float | None: The measured value, or None if it can't be computed.
        """
        raise NotImplementedError()


class EvaluationWidthCompute(EvaluationCompute):
    """Flags cores whose width deviates too far from the reference width.

    Args:
        config (CoreWidthCheckConfig): Tunable parameters for the core width check.
    """

    def __init__(self, config: CoreWidthCheckConfig):
        super().__init__(config)
        self.max_width_steps = config.max_width_steps
        self.relative_tolerance_steps = config.relative_tolerance_steps

    def _measure(self, detection: ImageMetadataProcessedCores) -> float | None:
        """Compute a core's width in pixels: its bounding box's vertical (y) extent.

        Args:
            detection (ImageMetadataProcessedCores): The processed image metadata to measure.

        Returns:
            float | None: Width in pixels, or None if no core was detected for this image.
        """
        if detection.core is None or detection.ruler is None or detection.ruler.px_per_unit is None:
            return None

        return (detection.core.bbox[3] - detection.core.bbox[1]) / detection.ruler.px_per_unit

    def _estimate_segments(
        self, depths: list[float], values: list[float | None]
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Split core widths into depth segments using dynamic-programming change-point detection.

        Args:
            depths (list[float]): Starting depth for each detection, matched by index to `values`.
            values (list[float | None]): Measured core widths, in detection order.

        Returns:
            tuple[list[tuple[float, float]], list[float]]: Segment boundaries as (start, end)
                depth pairs, and each segment's mean width, found by `DPCoreWidthEstimation`.
        """
        estimator = DPCoreWidthEstimation(max_k=self.max_width_steps, alpha=self.relative_tolerance_steps)
        estimator.fit(depths=depths, widths=values)
        return estimator._segments, estimator._references


class EvaluationLengthCompute(EvaluationCompute):
    """Flags cores whose length-to-depth ratio deviates too far from the reference ratio.

    Args:
        config (CoreLengthCheckConfig): Tunable parameters (relative_tolerance, min_samples,
            max_depth_range) for the core length check.
    """

    def __init__(self, config: CoreLengthCheckConfig):
        super().__init__(config)
        self.max_depth_range = config.max_depth_range

    def _measure(self, detection: ImageMetadataProcessedCores) -> float | None:
        """Compute a core's length-to-depth ratio, normalized by the ruler's px-per-unit scale.

        Args:
            detection (ImageMetadataProcessedCores): The processed image metadata to measure.

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


def evaluate_detections(
    detections: list[ImageMetadataProcessedCores], config: EvaluationConfig
) -> list[CoreCheckResult]:
    """Run the width and length checks and merge them into one result per file.

    Args:
        detections (list[ImageMetadataProcessedCores]): List of processed image metadata with detection results.
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
