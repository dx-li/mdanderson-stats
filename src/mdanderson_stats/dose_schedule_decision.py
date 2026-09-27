"""Safety-screened dose and administration-schedule allocation."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .dose_schedule_fit import DoseScheduleFit


def _freeze_bool(value: ArrayLike) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    return np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pair(value: object, name: str, shape: tuple[int, int]) -> tuple[int, int]:
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer (dose, schedule) pair")
    pair = (int(raw[0]), int(raw[1]))
    if not (0 <= pair[0] < shape[0] and 0 <= pair[1] < shape[1]):
        raise ValueError(f"{name} is outside the fitted dose/schedule grid")
    return pair


@dataclass(frozen=True)
class DoseScheduleDecision:
    """One allocation decision and the masks used to make it."""

    action: str
    pair: tuple[int, int] | None
    mean_risk: FloatArray
    overdose_probability: FloatArray
    acceptable: np.ndarray
    eligible: np.ndarray
    distance: float | None
    reason: str


def dose_schedule_decision(
    fit: DoseScheduleFit,
    treated: ArrayLike,
    *,
    toxicity_limit: float,
    upper_probability: float,
    target: float,
    current: tuple[int, int] | None = None,
    starting: tuple[int, int] = (0, 0),
    final: bool = False,
    escalation_rule: str = "coordinate_max",
) -> DoseScheduleDecision:
    """Choose a safe regimen nearest the target, subject to a no-skip rule.

    ``coordinate_max`` allows a candidate no more than one level above each
    maximum tried coordinate. ``observed_pair`` allows the union of one-step
    rectangles around previously tried dose/schedule pairs. Final selection
    removes the no-skip restriction. An empty design starts at ``starting``
    without a posterior safety screen, following the paper's first-patient rule.
    """
    if not isinstance(fit, DoseScheduleFit):
        raise ValueError("fit must be a DoseScheduleFit")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if escalation_rule not in ("coordinate_max", "observed_pair"):
        raise ValueError("escalation_rule must be coordinate_max or observed_pair")
    limit = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(upper_probability, "upper_probability")
    target_value = _scalar(target, "target")
    if not 0 <= limit <= 1 or not 0 < cutoff < 1 or not 0 <= target_value <= 1:
        raise ValueError("toxicity_limit/target must be in [0,1] and upper_probability in (0,1)")
    risks = np.asarray(fit.regimen_risk)
    if risks.ndim != 4 or risks.shape[0] < 1 or risks.shape[1] < 1:
        raise ValueError("fit regimen risks must have chain/draw/dose/schedule axes")
    if risks.shape[2] > 20 or risks.shape[3] > 20 or risks.size > 2_000_000:
        raise ValueError("fit risk grid exceeds the bounded dose/schedule allocation size")
    if np.any(~np.isfinite(risks)) or np.any((risks < 0) | (risks > 1)):
        raise ValueError("fit regimen risks must be finite probabilities")
    grid = risks.shape[2:]
    raw_treated = np.asarray(treated)
    if (
        raw_treated.shape != grid
        or raw_treated.dtype.kind not in "iu"
        or raw_treated.dtype.kind == "b"
    ):
        raise ValueError("treated must be an integer array matching the dose/schedule grid")
    if np.any(raw_treated < 0) or np.sum(raw_treated, dtype=object) > 10_000:
        raise ValueError("treated counts must be nonnegative and total at most 10,000")
    counts = raw_treated.astype(np.int64, copy=False)
    start = _pair(starting, "starting", grid)
    total = int(np.sum(counts, dtype=np.int64))
    if total == 0:
        if final:
            raise ValueError("final selection requires at least one treated patient")
        mean_risk = np.mean(risks, axis=(0, 1))
        overdose = np.mean(risks > limit, axis=(0, 1))
        acceptable = overdose < cutoff
        return DoseScheduleDecision(
            "start",
            start,
            _freeze(mean_risk),
            _freeze(overdose),
            _freeze_bool(acceptable),
            _freeze_bool(acceptable),
            None,
            "first patient follows the configured starting regimen",
        )
    if current is not None:
        _pair(current, "current", grid)

    mean_risk = np.mean(risks, axis=(0, 1))
    overdose = np.mean(risks > limit, axis=(0, 1))
    acceptable = overdose < cutoff
    eligible = acceptable.copy()
    if not final:
        tried = np.argwhere(counts > 0)
        if escalation_rule == "coordinate_max":
            upper = np.minimum(np.max(tried, axis=0) + 1, np.asarray(grid) - 1)
            eligible &= (np.indices(grid)[0] <= upper[0]) & (np.indices(grid)[1] <= upper[1])
        else:
            allowed = np.zeros(grid, dtype=bool)
            for dose, schedule in tried:
                allowed[: min(grid[0], dose + 2), : min(grid[1], schedule + 2)] = True
            eligible &= allowed
    if not np.any(eligible):
        return DoseScheduleDecision(
            "stop",
            None,
            _freeze(mean_risk),
            _freeze(overdose),
            _freeze_bool(acceptable),
            _freeze_bool(eligible),
            None,
            "no candidate satisfies the posterior overdose cutoff and allocation constraints",
        )
    distances = np.abs(mean_risk - target_value)
    distances = np.where(eligible, distances, np.inf)
    flat_index = int(np.argmin(distances))
    chosen_raw = np.unravel_index(flat_index, grid)
    chosen = (int(chosen_raw[0]), int(chosen_raw[1]))
    return DoseScheduleDecision(
        "select" if final else "treat",
        chosen,
        _freeze(mean_risk),
        _freeze(overdose),
        _freeze_bool(acceptable),
        _freeze_bool(eligible),
        float(distances[chosen]),
        "nearest acceptable regimen to target, with lexicographic tie breaking",
    )
