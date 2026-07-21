"""Configuration and result models for the per-core consistency checks (width, length, ...)."""

from dataclasses import dataclass, field


@dataclass
class CoreCheckConfig:
    """Common tunable parameters shared by all per-core consistency checks (width, length, ...)."""

    relative_tolerance: float = 0.1  # flag if the deviation from the expected/reference value exceeds this
    min_samples: int = 5  # below this, skip the check (median/ratio unreliable with too few points)


@dataclass
class CoreWidthCheckConfig(CoreCheckConfig):
    """Tunable parameters for the core width check evaluation."""

    relative_tolerance: float = 0.25  # flag if |width - folder_median| / folder_median exceeds this


@dataclass
class CoreLengthCheckConfig(CoreCheckConfig):
    """Tunable parameters for the core length check evaluation."""

    relative_tolerance: float = 0.05  # ~5% buffer


@dataclass
class EvaluationConfig:
    """Tunable parameters for the evaluation step."""

    core_width: CoreWidthCheckConfig = field(default_factory=CoreWidthCheckConfig)
    core_length: CoreLengthCheckConfig = field(default_factory=CoreLengthCheckConfig)


@dataclass
class CoreCheckOutcome:
    """Common pass/fail verdict fields shared by all per-core consistency checks (width, length, ...)."""

    passed: bool  # whether the core passed the check (True = within tolerance, False = flagged)
    deviation: float  # relative deviation from the expected/reference value


@dataclass
class CoreWidthCheckResult(CoreCheckOutcome):
    """Results of the core width check evaluation."""

    width: float  # width of the core in pixels
    folder_median_width: float  # median width of cores in the folder


@dataclass
class CoreLengthCheckResult(CoreCheckOutcome):
    """Results of the core length check evaluation."""

    length_px: float  # length of the core in pixels
    expected_length_px: float  # (depth_end - depth_start) * folder_ratio_px_per_m
    folder_ratio_px_per_m: float  # median px-per-metre ratio for this folder


@dataclass
class CoreCheckResult:
    """All per-core consistency check results for a single file.

    width/length are None when the corresponding check was skipped for this file
    (e.g. too few detections for a reliable reference, see CoreCheckConfig.min_samples).
    """

    filename: str  # name of the image file
    width: CoreWidthCheckResult | None
    length: CoreLengthCheckResult | None
