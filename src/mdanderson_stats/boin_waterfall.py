"""Planning helpers for the BOIN-waterfall combination design.

The waterfall design partitions a ``J`` by ``K`` dose matrix into an initial
staircase (the first column followed by the top row), then one row slice at a
time.  This module plans the next slice from completed dose counts; conduct
within a slice is delegated to :class:`~mdanderson_stats.boin.BOINDesign`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincc

from .boin import BOINDesign, _owned
from ._validation import scalar

DoseCombination = tuple[int, int]
BoolMatrix = NDArray[np.bool_]


def _counts(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a matrix with at least 2 rows and 2 columns")
    if result.shape[0] > result.shape[1] or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix with rows <= columns")
    if np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return result


def _validate_counts(patients: ArrayLike, toxicities: ArrayLike) -> tuple[NDArray, NDArray]:
    n = _counts(patients, "patients")
    y = _counts(toxicities, "toxicities")
    if n.shape != y.shape:
        raise ValueError("patients and toxicities must have matching dimensions")
    if np.any(y > n):
        raise ValueError("toxicities cannot exceed patients")
    return n, y


def _pava(values: NDArray, weights: NDArray) -> NDArray:
    levels: list[float] = []
    weights_out: list[float] = []
    sizes: list[int] = []
    for value, weight in zip(values, weights, strict=True):
        levels.append(float(value))
        weights_out.append(float(weight))
        sizes.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            total = weights_out[-2] + weights_out[-1]
            levels[-2] = (levels[-2] * weights_out[-2] + levels[-1] * weights_out[-1]) / total
            weights_out[-2] = total
            sizes[-2] += sizes[-1]
            levels.pop(); weights_out.pop(); sizes.pop()
    return np.repeat(np.asarray(levels), sizes)


def _initial_space(rows: int, columns: int) -> tuple[DoseCombination, ...]:
    return tuple([(i, 1) for i in range(1, rows + 1)] + [(rows, j) for j in range(2, columns + 1)])


def _row_space(row: int, columns: int) -> tuple[DoseCombination, ...]:
    return tuple((row, j) for j in range(2, columns + 1))


def _space_for_subtrial(index: int, rows: int, columns: int) -> tuple[DoseCombination, ...]:
    if index == rows:
        return _initial_space(rows, columns)
    return _row_space(index, columns)


@dataclass(frozen=True)
class WaterfallPlan:
    """Result of planning the next BOIN-waterfall subtrial.

    Dose labels are one-based. ``None`` means that all subtrials have been
    completed or the current subtrial stopped for safety. ``candidate_mtd`` is
    the isotonic candidate selected from the most recently active slice.
    """

    next_subtrial: tuple[DoseCombination, ...] | None
    starting_dose: DoseCombination | None
    current_subtrial: int | None
    candidate_mtd: DoseCombination | None
    escalation: bool | None
    eliminated: BoolMatrix
    action: str

    @property
    def dose_search_space(self) -> tuple[DoseCombination, ...] | None:
        """Descriptive alias for ``next_subtrial``."""

        return self.next_subtrial


def _candidate(
    design: BOINDesign,
    n: NDArray,
    y: NDArray,
    space: tuple[DoseCombination, ...],
    *,
    extrasafe: bool,
    cutoff: float,
    offset: float,
) -> tuple[DoseCombination | None, bool | None, BoolMatrix, str]:
    posterior = betaincc(y + 1, n - y + 1, design.target)
    eliminated = np.zeros(n.shape, dtype=bool)
    for i, j in zip(*np.where((n >= 3) & (posterior > cutoff)), strict=True):
        eliminated[i:, j:] = True
    if extrasafe and n[0, 0] >= 3 and posterior[0, 0] > cutoff - offset:
        eliminated[:, :] = True
    if eliminated[0, 0]:
        return None, None, _owned(eliminated), "stop_safety"

    active = [(i - 1, j - 1) for i, j in space if n[i - 1, j - 1] > 0 and not eliminated[i - 1, j - 1]]
    if not active:
        return None, None, _owned(eliminated), "stop_no_data"
    observed = np.asarray([y[i, j] for i, j in active], dtype=float)
    treated = np.asarray([n[i, j] for i, j in active], dtype=float)
    estimate = (observed + 0.05) / (treated + 0.1)
    variance = (observed + 0.05) * (treated - observed + 0.05) / ((treated + 0.1) ** 2 * (treated + 1.1))
    fitted = _pava(estimate, 1.0 / variance) + np.arange(1, len(active) + 1) * 1e-10
    chosen = int(np.argmin(np.abs(fitted - design.target)))
    candidate = (active[chosen][0] + 1, active[chosen][1] + 1)
    boundary = design.boundary_table(max(150, int(np.max(treated)))).escalate_max
    count = int(treated[chosen])
    escalate = int(observed[chosen]) <= int(boundary[count - 1])
    return candidate, bool(escalate), _owned(eliminated), "continue"


def next_subtrial(
    target: float,
    patients: ArrayLike | None = None,
    toxicities: ArrayLike | None = None,
    *,
    npts: ArrayLike | None = None,
    ntox: ArrayLike | None = None,
    safe_probability: float | None = None,
    toxic_probability: float | None = None,
    elimination_probability: float = 0.95,
    extra_safe: bool = False,
    safety_offset: float = 0.05,
    p_saf: float | None = None,
    p_tox: float | None = None,
    cutoff_eli: float | None = None,
    extrasafe: bool | None = None,
    offset: float | None = None,
) -> WaterfallPlan:
    """Determine the next dose-searching slice and its starting dose.

    This is the Python equivalent of BOIN's ``next.subtrial``.  The highest
    indexed slice containing data is treated as current.  Its candidate is
    estimated by weighted one-dimensional isotonic regression; the next slice
    is one row lower and starts one column to the right of that candidate.
    """

    if patients is None:
        patients = npts
    elif npts is not None:
        raise ValueError("provide patients or npts, not both")
    if toxicities is None:
        toxicities = ntox
    elif ntox is not None:
        raise ValueError("provide toxicities or ntox, not both")
    if patients is None or toxicities is None:
        raise TypeError("patients and toxicities are required")
    if p_saf is not None:
        if safe_probability is not None:
            raise ValueError("provide safe_probability or p_saf, not both")
        safe_probability = p_saf
    if p_tox is not None:
        if toxic_probability is not None:
            raise ValueError("provide toxic_probability or p_tox, not both")
        toxic_probability = p_tox
    if cutoff_eli is not None:
        if elimination_probability != 0.95:
            raise ValueError("provide elimination_probability or cutoff_eli, not both")
        elimination_probability = cutoff_eli
    if extrasafe is not None:
        if extra_safe:
            raise ValueError("provide extra_safe or extrasafe, not both")
        extra_safe = extrasafe
    if offset is not None:
        if safety_offset != 0.05:
            raise ValueError("provide safety_offset or offset, not both")
        safety_offset = offset
    n, y = _validate_counts(patients, toxicities)
    rows, columns = n.shape
    design = BOINDesign(
        target=scalar(target, "target"),
        safe_probability=safe_probability,
        toxic_probability=toxic_probability,
        elimination_probability=elimination_probability,
        extra_safe=extra_safe,
        safety_offset=safety_offset,
    )
    occupied = [i for i in range(rows, 0, -1) if np.any(n[np.asarray(_space_for_subtrial(i, rows, columns)).T[0] - 1, np.asarray(_space_for_subtrial(i, rows, columns)).T[1] - 1] > 0)]
    if not occupied:
        return WaterfallPlan(None, None, None, None, None, _owned(np.zeros_like(n, dtype=bool)), "no_data")
    current = occupied[0]
    space = _space_for_subtrial(current, rows, columns)
    candidate, escalation, eliminated, action = _candidate(
        design, n, y, space, extrasafe=extra_safe, cutoff=elimination_probability, offset=safety_offset
    )
    if current == 1 or candidate is None:
        return WaterfallPlan(None, None, current, candidate, escalation, eliminated, action if current != 1 else "complete")
    next_row = max(1, candidate[0] - 1)
    next_col = min(columns, candidate[1] + 1)
    next_space = _row_space(next_row, columns)
    return WaterfallPlan(next_space, (next_row, next_col), current, candidate, escalation, eliminated, "next_subtrial")


@dataclass(frozen=True)
class BOINWaterfall:
    """Configured planner exposing ``next_subtrial`` for repeated updates."""

    target: float = 0.25
    safe_probability: float | None = None
    toxic_probability: float | None = None
    elimination_probability: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05

    def next_subtrial(self, patients: ArrayLike, toxicities: ArrayLike) -> WaterfallPlan:
        return next_subtrial(
            self.target,
            patients,
            toxicities,
            safe_probability=self.safe_probability,
            toxic_probability=self.toxic_probability,
            elimination_probability=self.elimination_probability,
            extra_safe=self.extra_safe,
            safety_offset=self.safety_offset,
        )
