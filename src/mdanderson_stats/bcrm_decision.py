"""Explicit single-outcome bCRM dose-selection and stopping rules."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite, scalar
from .beta_binomial import _owned

_MAX_ALLOCATION_VALUES = 200_000


@dataclass(frozen=True)
class BCRMDecision:
    """Dose decision using zero-based indices; ``None`` means stop treatment."""

    target_index: int
    next_index: int | None
    stop: bool
    reason: str
    target_attainable: bool


def _integer(value: int, name: str) -> int:
    numeric = scalar(value, name)
    if numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _target_index(probabilities: np.ndarray, target: float, selection: str) -> tuple[int, bool]:
    if selection == "below":
        eligible = np.flatnonzero(probabilities <= target)
        return (int(eligible[-1]), True) if eligible.size else (0, False)
    if selection == "above":
        eligible = np.flatnonzero(probabilities >= target)
        return (int(eligible[0]), True) if eligible.size else (probabilities.size - 1, False)
    if selection == "nearest":
        distances = np.abs(probabilities - target)
        minimum = float(np.min(distances))
        tied = np.flatnonzero(distances <= minimum + 8 * np.finfo(float).eps)
        return int(tied[0]), True
    raise ValueError("selection must be 'below', 'nearest', or 'above'")


def bcrm_extreme_allocation_probability(
    target_fraction: ArrayLike,
    allocated_fraction: ArrayLike,
    *,
    correction: float = 2.0,
) -> np.ndarray:
    """Return the guide-defined probability for extra extreme-dose allocation.

    For target fraction ``p_T`` and the fraction already allocated ``p_o``,
    the guide uses ``p_T ** (1 + correction * (p_o - p_T))`` and constrains
    the result to ``[0.1, 0.5]``. Fractions may be scalars or broadcast
    one-dimensional vectors; target fractions lie in (0, 1), allocated
    fractions in [0, 1]. Correction is a finite nonnegative scalar. This is a
    randomization probability, not an efficacy estimate or a treatment-success
    probability.
    """
    target = _allocation_fraction(target_fraction, "target_fraction")
    allocated = _allocation_fraction(allocated_fraction, "allocated_fraction")
    if isinstance(correction, np.ndarray):
        if correction.ndim != 0:
            raise ValueError("correction must be a scalar")
        if np.iscomplexobj(correction):
            raise ValueError("correction must be real-valued")
    elif not np.isscalar(correction):
        raise ValueError("correction must be a scalar")
    elif np.iscomplexobj(correction):
        raise ValueError("correction must be real-valued")
    gamma = scalar(float(np.asarray(correction, dtype=np.float64)), "correction")
    if np.any((target <= 0) | (target >= 1)):
        raise ValueError("target_fraction must lie strictly inside (0, 1)")
    if np.any((allocated < 0) | (allocated > 1)):
        raise ValueError("allocated_fraction must lie in [0, 1]")
    if gamma < 0:
        raise ValueError("correction must be nonnegative")
    if (
        target.ndim == allocated.ndim == 1
        and target.size not in (1, allocated.size)
        and allocated.size != 1
    ):
        raise ValueError("fraction vectors must have equal lengths or be scalar")
    target, allocated = np.broadcast_arrays(target, allocated)
    if target.size > _MAX_ALLOCATION_VALUES:
        raise ValueError(f"broadcast result exceeds {_MAX_ALLOCATION_VALUES} values")
    exponent = 1.0 + gamma * (allocated - target)
    log_target = np.log(target)
    log_lower, log_upper = np.log(0.1), np.log(0.5)
    # Since log_target < 0, threshold comparisons avoid multiplying an
    # extreme finite exponent by a large-magnitude log target.
    lower_exponent = log_lower / log_target
    upper_exponent = log_upper / log_target
    low = exponent >= lower_exponent
    high = exponent <= upper_exponent
    result = np.full(target.shape, 0.5, dtype=np.float64)
    result[low] = 0.1
    middle = ~(low | high)
    result[middle] = np.exp(exponent[middle] * log_target[middle])
    return _owned(result)


def _allocation_fraction(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim not in (0, 1) or value.size > _MAX_ALLOCATION_VALUES:
            raise ValueError(
                f"{name} must be a scalar or vector of at most {_MAX_ALLOCATION_VALUES}"
            )
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
    elif isinstance(value, Sequence):
        if len(value) > _MAX_ALLOCATION_VALUES:
            raise ValueError(f"{name} exceeds {_MAX_ALLOCATION_VALUES} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        if any(np.iscomplexobj(item) for item in value):
            raise ValueError(f"{name} must be real-valued")
    elif np.isscalar(value):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
    else:
        raise TypeError(
            f"{name} must be a scalar, NumPy array, or bounded one-dimensional sequence"
        )
    array = finite(value, name)
    if array.ndim not in (0, 1):
        raise ValueError(f"{name} must be a scalar or one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    return array


def bcrm_decision(
    probabilities: ArrayLike,
    subjects: ArrayLike,
    *,
    target: float,
    max_subjects: int,
    cohort_size: int = 2,
    min_subjects: int = 0,
    stop_at_target: int = 6,
    selection: str = "below",
    start_index: int = 0,
    max_increment: int = 1,
) -> BCRMDecision:
    """Choose a dose and apply cohort, escalation and stopping constraints.

    Probabilities may be posterior predictive means or another explicitly
    chosen single-outcome estimate. Dose indices are zero-based. An empty
    history returns ``start_index``. Escalation is capped relative to the
    highest dose ever tried, not the most recently treated dose.
    """
    raw_probabilities, raw_subjects = np.asarray(probabilities), np.asarray(subjects)
    if raw_probabilities.ndim != 1 or not 1 <= raw_probabilities.size <= 100:
        raise ValueError("probabilities must be one-dimensional with 1 to 100 doses")
    if raw_subjects.shape != raw_probabilities.shape:
        raise ValueError("subjects must match the probability vector shape")
    if selection not in {"below", "nearest", "above"}:
        raise ValueError("selection must be 'below', 'nearest', or 'above'")
    estimates = finite(raw_probabilities, "probabilities")
    if np.any((estimates < 0) | (estimates > 1)) or np.any(np.diff(estimates) < 0):
        raise ValueError("probabilities must be nondecreasing and lie in [0, 1]")
    allocations = count(raw_subjects, "subjects")
    if np.sum(allocations) > 10_000:
        raise ValueError("total subjects must be <= 10000")

    target_value = scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between zero and one")
    cohort = _integer(cohort_size, "cohort_size")
    maximum = _integer(max_subjects, "max_subjects")
    minimum = _integer(min_subjects, "min_subjects")
    stop_count = _integer(stop_at_target, "stop_at_target")
    start = _integer(start_index, "start_index")
    increment = _integer(max_increment, "max_increment")
    if cohort <= 0:
        raise ValueError("cohort_size must be positive")
    if maximum <= 0 or maximum > 10_000 or maximum % cohort:
        raise ValueError("max_subjects must be a positive cohort multiple <= 10000")
    if minimum < 0 or minimum > maximum or minimum % cohort:
        raise ValueError("min_subjects must be a cohort multiple from zero through max_subjects")
    if stop_count <= 0:
        raise ValueError("stop_at_target must be positive")
    if increment < 0:
        raise ValueError("max_increment must be nonnegative")
    if not 0 <= start < estimates.size:
        raise ValueError("start_index must identify a dose in the probability vector")
    if np.any(allocations % cohort):
        raise ValueError("each dose's subject count must be a cohort multiple")
    total = int(np.sum(allocations))
    if total > maximum:
        raise ValueError("total subjects must not exceed max_subjects")

    target_dose, attainable = _target_index(estimates, target_value, selection)
    if total == 0:
        next_dose = start
    else:
        highest_tried = int(np.flatnonzero(allocations > 0)[-1])
        next_dose = min(target_dose, highest_tried + increment)

    if total >= maximum:
        return BCRMDecision(target_dose, None, True, "max_subjects", attainable)
    if total >= minimum and allocations[target_dose] >= stop_count:
        return BCRMDecision(target_dose, None, True, "target_dose_limit", attainable)
    return BCRMDecision(target_dose, next_dose, False, "continue", attainable)
