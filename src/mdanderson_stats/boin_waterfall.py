"""Planning helpers for the BOIN-waterfall combination design.

The waterfall design partitions a ``J`` by ``K`` dose matrix into an initial
staircase (the first column followed by the last row), then one row slice at a
time.  This module plans the next slice from completed dose counts; conduct
within a slice is delegated to :class:`~mdanderson_stats.boin.BOINDesign`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betaincc

from ._validation import count, scalar
from .boin import BOINDesign, _owned

DoseCombination = tuple[int, int]
BoolMatrix = NDArray[np.bool_]


def _counts(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = count(value, name)
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
    if n.sum() > 1000:
        raise ValueError("total patients must not exceed 1000")
    return n, y


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
    extra_safe: bool,
    cutoff: float,
    offset: float,
) -> tuple[DoseCombination | None, bool | None, BoolMatrix, str]:
    posterior = betaincc(y + 1, n - y + 1, design.target)
    eliminated = np.zeros(n.shape, dtype=bool)
    for i, j in zip(*np.where((n >= 3) & (posterior > cutoff)), strict=True):
        eliminated[i:, j:] = True
    # The R helper applies elimination in the one-dimensional order of the
    # active slice.  In particular, extrasafe is checked at that slice's first
    # dose, rather than at matrix cell (1, 1).
    slice_eliminated = np.zeros(len(space), dtype=bool)
    for index, (i, j) in enumerate(space):
        if n[i - 1, j - 1] >= 3 and posterior[i - 1, j - 1] > cutoff:
            slice_eliminated[index:] = True
            break
    first_i, first_j = space[0]
    if (
        extra_safe
        and n[first_i - 1, first_j - 1] >= 3
        and posterior[first_i - 1, first_j - 1] > cutoff - offset
    ):
        slice_eliminated[:] = True
    for (i, j), excluded in zip(space, slice_eliminated, strict=True):
        if excluded:
            eliminated[i - 1 :, j - 1 :] = True
    if slice_eliminated[0]:
        return None, None, _owned(eliminated), "stop_safety"

    active = [
        (i - 1, j - 1)
        for (i, j), excluded in zip(space, slice_eliminated, strict=True)
        if n[i - 1, j - 1] > 0 and not excluded
    ]
    if not active:
        return None, None, _owned(eliminated), "stop_no_data"
    observed = np.asarray([y[i, j] for i, j in active], dtype=float)
    treated = np.asarray([n[i, j] for i, j in active], dtype=float)
    estimate = (observed + 0.05) / (treated + 0.1)
    variance = (
        (observed + 0.05) * (treated - observed + 0.05) / ((treated + 0.1) ** 2 * (treated + 1.1))
    )
    fitted = isotonic_regression(estimate, weights=1.0 / variance).x
    fitted = fitted + np.arange(1, len(active) + 1) * 1e-10
    chosen = int(np.argmin(np.abs(fitted - design.target)))
    candidate = (active[chosen][0] + 1, active[chosen][1] + 1)
    count = int(treated[chosen])
    boundary = design.boundary_table(max(1, count)).escalate_max
    escalate = int(observed[chosen]) <= int(boundary[count - 1])
    return candidate, bool(escalate), _owned(eliminated), "continue"


def next_subtrial(
    target: float,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    safe_probability: float | None = None,
    toxic_probability: float | None = None,
    elimination_probability: float = 0.95,
    extra_safe: bool = False,
    safety_offset: float = 0.05,
) -> WaterfallPlan:
    """Determine the next dose-searching slice and its starting dose.

    This is the Python equivalent of BOIN's ``next.subtrial``.  The lowest
    indexed slice containing data is treated as current.  Its candidate is
    estimated by weighted one-dimensional isotonic regression; the next slice
    is one row lower and starts one column to the right of that candidate.
    """

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
    occupied = []
    for i in range(1, rows + 1):
        space_i = _space_for_subtrial(i, rows, columns)
        indices = np.asarray(space_i, dtype=int) - 1
        if np.any(n[indices[:, 0], indices[:, 1]] > 0):
            occupied.append(i)
    if not occupied:
        return WaterfallPlan(
            None, None, None, None, None, _owned(np.zeros_like(n, dtype=bool)), "no_data"
        )
    current = occupied[0]
    space = _space_for_subtrial(current, rows, columns)
    candidate, escalation, eliminated, action = _candidate(
        design,
        n,
        y,
        space,
        extra_safe=extra_safe,
        cutoff=elimination_probability,
        offset=safety_offset,
    )
    if current == 1 or candidate is None:
        return WaterfallPlan(
            None,
            None,
            current,
            candidate,
            escalation,
            eliminated,
            action if action != "continue" else "complete",
        )
    next_row = max(1, candidate[0] - 1)
    next_col = min(columns, candidate[1] + 1)
    next_space = _row_space(next_row, columns)
    return WaterfallPlan(
        next_space,
        (next_row, next_col),
        current,
        candidate,
        escalation,
        eliminated,
        "next_subtrial",
    )


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
