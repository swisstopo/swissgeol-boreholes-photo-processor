"""Configuration and result models for the per-core consistency checks (width, length, ...)."""

from dataclasses import dataclass, field


@dataclass
class CoreCheckConfig:
    """Common tunable parameters shared by all per-core consistency checks (width, length, ...)."""

    relative_tolerance: float  # flag if the deviation from the expected/reference value exceeds this
    min_samples: int = 5  # below this, skip the check (median/ratio unreliable with too few points)


@dataclass
class CoreWidthCheckConfig(CoreCheckConfig):
    """Tunable parameters for the core width check evaluation."""

    relative_tolerance: float = 0.25  # flag if |width - batch_median| / batch_median exceeds this


@dataclass
class CoreLengthCheckConfig(CoreCheckConfig):
    """Tunable parameters for the core length check evaluation."""

    relative_tolerance: float = 0.05  # ~5% buffer on the length-to-depth ratio vs. the batch median
    max_depth_range: float = 1.00  # Cap height scale to 1 meter (no image with more than 1m core)


@dataclass
class EvaluationConfig:
    """Tunable parameters for the evaluation step."""

    core_width: CoreWidthCheckConfig = field(default_factory=CoreWidthCheckConfig)
    core_length: CoreLengthCheckConfig = field(default_factory=CoreLengthCheckConfig)


@dataclass
class CoreValueCheckResult:
    """Common pass/fail verdict fields shared by all per-core consistency checks (width, length, ...)."""

    passed: bool  # whether the core passed the check (True = within tolerance, False = flagged)
    relative_error: float  # (measure - reference) / reference
    measure: float  # value computed for this detection
    reference: float  # group median value


@dataclass
class CoreCheckResult:
    """All per-core consistency check results for a single file.

    width/length are None when the corresponding check was skipped for this file
    (e.g. too few detections for a reliable reference, see CoreCheckConfig.min_samples).
    """

    filename: str  # name of the image file
    width: CoreValueCheckResult | None
    length: CoreValueCheckResult | None
