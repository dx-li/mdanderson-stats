"""Mathematical core of the MD Anderson TTEConduct stopping-boundary tool."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ._validation import count, scalar
from .inequality import inequality_probability
from .parameter_distribution import ParameterDistribution


@dataclass(frozen=True)
class TTEConductDesign:
    """Fixed historical and experimental inverse-gamma design parameters."""

    alpha_standard: float
    beta_standard: float
    alpha_experimental: float
    beta_experimental: float
    delta: float
    cutoff: float
    max_patients: int
    absolute_tolerance: float
    max_total_time: float


@dataclass(frozen=True)
class TTEConductMonitor:
    patients: int
    events: int
    total_time: float
    probability: float
    probability_error: float
    stop_for_futility: bool
    stop_for_maximum: bool
    stop_accrual: bool


@dataclass(frozen=True)
class TTEConductBoundary:
    events: int
    minimum_total_time: float
    probability: float
    probability_error: float
    residual: float
    beyond_cap: bool


@dataclass(frozen=True)
class TTEConductBoundaryTable:
    boundaries: tuple[TTEConductBoundary, ...]


def _positive(value: float, name: str) -> float:
    value = scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def tteconduct_design(
    alpha_standard: float,
    beta_standard: float,
    alpha_experimental: float,
    beta_experimental: float,
    delta: float,
    cutoff: float,
    max_patients: int,
    *,
    max_total_time: float,
    absolute_tolerance: float = 1e-9,
) -> TTEConductDesign:
    """Validate and construct a TTEConduct exponential/inverse-gamma design.

    All time quantities use the caller's single time unit.  ``max_total_time``
    is required only to make boundary searches bounded; it is not a clinical
    conversion or an implicit day/month assumption.
    """
    a_s, b_s = (
        _positive(alpha_standard, "alpha_standard"),
        _positive(beta_standard, "beta_standard"),
    )
    a_e, b_e = (
        _positive(alpha_experimental, "alpha_experimental"),
        _positive(beta_experimental, "beta_experimental"),
    )
    delta = scalar(delta, "delta")
    if delta < 0:
        raise ValueError("delta must be nonnegative")
    cutoff = scalar(cutoff, "cutoff")
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must be strictly between zero and one")
    if isinstance(max_patients, bool) or np.ndim(max_patients) != 0:
        raise ValueError("max_patients must be an integer in [1,1000]")
    max_patients_value = scalar(max_patients, "max_patients")
    if (
        max_patients_value < 1
        or max_patients_value > 1000
        or int(max_patients_value) != max_patients_value
    ):
        raise ValueError("max_patients must be an integer in [1,1000]")
    tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12,1e-3]")
    cap = _positive(max_total_time, "max_total_time")
    return TTEConductDesign(
        a_s, b_s, a_e, b_e, delta, cutoff, int(max_patients_value), tolerance, cap
    )


def _probability(design: TTEConductDesign, events: int, total_time: float) -> tuple[float, float]:
    if total_time < 0 or not np.isfinite(total_time):
        raise ValueError("total_time must be finite and nonnegative")
    shape = design.alpha_experimental + events
    scale = design.beta_experimental + total_time
    if not np.isfinite(shape) or not np.isfinite(scale):
        raise ArithmeticError("inverse-gamma posterior parameters overflow")
    experimental = ParameterDistribution("inverse_gamma", shape, scale)
    standard = ParameterDistribution("inverse_gamma", design.alpha_standard, design.beta_standard)
    try:
        result = inequality_probability(
            experimental,
            standard,
            delta=design.delta,
            absolute_tolerance=design.absolute_tolerance,
        )
    except ArithmeticError:
        # QUADPACK can occasionally miss its subdivision target at an isolated
        # root probe. Retry at a boundedly looser tolerance and report that
        # resulting error rather than fabricating a converged value.
        result = inequality_probability(
            experimental,
            standard,
            delta=design.delta,
            absolute_tolerance=min(1e-3, design.absolute_tolerance * 10),
        )
    if not np.isfinite(result.x_greater) or not np.isfinite(result.absolute_error):
        raise ArithmeticError("inverse-gamma stopping probability is unresolved")
    return float(result.x_greater), float(result.absolute_error)


def tteconduct_monitor(
    design: TTEConductDesign,
    patients: int,
    events: int,
    total_time: float,
) -> TTEConductMonitor:
    """Evaluate the strict futility rule and maximum-patient stop condition."""
    if not isinstance(design, TTEConductDesign):
        raise TypeError("design must be a TTEConductDesign")
    if isinstance(patients, bool) or np.ndim(patients) != 0:
        raise ValueError("patients must be an integer")
    patients_value = scalar(patients, "patients")
    if isinstance(events, bool):
        raise ValueError("events must be an integer")
    events_value = scalar(events, "events")
    if (
        patients_value < 0
        or patients_value > design.max_patients
        or int(patients_value) != patients_value
    ):
        raise ValueError("patients must be an integer in [0,max_patients]")
    if events_value < 0 or events_value > patients_value or int(events_value) != events_value:
        raise ValueError("events must be an integer in [0,patients]")
    time = scalar(total_time, "total_time")
    if (
        time < 0
        or not np.isfinite(time)
        or (events_value > 0 and time <= 0)
        or (patients_value == 0 and time > 0)
    ):
        raise ValueError(
            "total_time must be nonnegative, positive with events, and zero before enrollment"
        )
    probability, error = _probability(design, int(events_value), time)
    futility = probability < design.cutoff
    maximum = int(patients_value) >= design.max_patients
    return TTEConductMonitor(
        int(patients_value),
        int(events_value),
        time,
        probability,
        error,
        futility,
        maximum,
        bool(futility or maximum),
    )


def _boundary(design: TTEConductDesign, events: int) -> TTEConductBoundary:
    at_zero, error_zero = _probability(design, events, 0.0)
    if at_zero >= design.cutoff:
        return TTEConductBoundary(events, 0.0, at_zero, error_zero, at_zero - design.cutoff, False)
    # Grow a finite time bracket rather than forming cap / beta_E, which can
    # overflow even when both input quantities are representable.
    high = min(design.beta_experimental, design.max_total_time)
    probability, error = _probability(design, events, high)
    for _ in range(2048):
        if probability >= design.cutoff:
            break
        if high >= design.max_total_time:
            return TTEConductBoundary(
                events, np.inf, probability, error, probability - design.cutoff, True
            )
        high = design.max_total_time if high >= design.max_total_time / 2 else high * 2
        probability, error = _probability(design, events, high)
    else:
        raise ArithmeticError("could not bracket TTEConduct boundary")

    # Solve on a dimensionless fraction of the discovered bracket. The
    # bracket is scaled locally, preserving unit invariance and root precision.
    low = 0.0
    bracket = high

    def objective(normalized_time: float) -> float:
        probability, _ = _probability(design, events, bracket * normalized_time)
        return probability - design.cutoff

    root = brentq(objective, low, 1.0, xtol=1e-14, rtol=1e-14, maxiter=100)
    high = bracket * root
    probability, error = _probability(design, events, high)
    return TTEConductBoundary(events, high, probability, error, probability - design.cutoff, False)


def tteconduct_boundary_table(
    design: TTEConductDesign,
    events: int | ArrayLike | None = None,
) -> TTEConductBoundaryTable:
    """Return minimum total time on test required to continue after each event count."""
    if not isinstance(design, TTEConductDesign):
        raise TypeError("design must be a TTEConductDesign")
    values: np.ndarray
    if events is None:
        values = np.arange(1, design.max_patients + 1, dtype=np.int64)
    else:
        values = np.asarray(events)
        if values.dtype.kind == "b":
            raise ValueError("events must contain integers")
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or len(values) == 0 or len(values) > 1000:
            raise ValueError("events must be a scalar or vector of at most 1000 values")
        values = count(values, "events")
    if np.any(values < 1) or np.any(values > design.max_patients):
        raise ValueError("events must lie in [1,max_patients]")
    return TTEConductBoundaryTable(tuple(_boundary(design, int(value)) for value in values))
