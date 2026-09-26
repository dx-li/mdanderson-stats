"""Explicit single-outcome bCRM dose-selection and stopping rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite, scalar


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
