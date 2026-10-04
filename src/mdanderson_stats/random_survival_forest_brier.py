"""Source-convention OOB Brier score and integrated CRPS for survival forests."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .random_survival_forest import (
    _MAX_OOB_WORK,
    RandomSurvivalForestFit,
    _forest_data,
    _forest_fingerprint,
    _integer,
)

_MAX_BRIER_CELLS = 2_000_000


@dataclass(frozen=True)
class RandomSurvivalForestOOBBrier:
    """Per-row OOB Brier contributions and their time-integrated summaries.

    Rows without OOB contributors remain NaN in ``brier`` and are omitted from
    each time-point mean. ``censor_survival`` follows the pinned RF-SRC
    censoring convention, including its grid projection.
    """

    time_grid: FloatArray
    brier: FloatArray
    score: FloatArray
    censor_survival: FloatArray
    valid_row_count: int
    crps: float
    crps_standardized: float


def _censor_survival(time: FloatArray, event: FloatArray, grid: FloatArray) -> FloatArray:
    """Return exp(-Nelson-Aalen censor hazard), projected by source sIndex."""
    censor_times, censor_counts = np.unique(time[event == 0], return_counts=True)
    result = np.ones(grid.size, dtype=np.float64)
    if censor_times.size == 0:
        return result

    sorted_time = np.sort(time)
    risk = time.size - np.searchsorted(sorted_time, censor_times, side="left")
    increments = censor_counts / risk
    cumulative = np.cumsum(increments, dtype=np.float64)
    mapped = np.searchsorted(censor_times, grid, side="right")
    affected = mapped > 0
    result[affected] = np.exp(-cumulative[mapped[affected] - 1])
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError("censoring survival is outside the representable positive range")
    return result


def _trapz_over_source_grid(grid: FloatArray, score: FloatArray) -> tuple[float, float]:
    """Integrate on the supplied grid without adding an artificial time zero."""
    if np.any(~np.isfinite(score)):
        return float("nan"), float("nan")
    if grid.size == 1:
        area = 0.0
        scale = float(grid[0])
        return area, float("nan") if scale == 0 else area / scale
    scale = float(grid[-1])
    if scale == 0:
        return 0.0, float("nan")
    normalized_width = np.diff(grid) / scale
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_area = float(np.sum(normalized_width * (score[:-1] / 2.0 + score[1:] / 2.0)))
        area = normalized_area * scale
    if not np.isfinite(normalized_area) or not np.isfinite(area):
        raise ArithmeticError("integrated Brier score exceeds floating-point range")
    return area, normalized_area


def random_survival_forest_oob_brier_score(
    fit: RandomSurvivalForestFit,
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    max_cells: int = _MAX_BRIER_CELLS,
    max_work: int = _MAX_OOB_WORK,
) -> RandomSurvivalForestOOBBrier:
    """Compute the pinned RF-SRC OOB IPCW Brier score and integrated CRPS.

    This API deliberately supports the complete training population only:
    native subset handling in the cached helper is ambiguous. The censoring
    distribution uses all supplied training outcomes, while rows with no OOB
    forest prediction are omitted from per-time score averages.
    """
    if not isinstance(fit, RandomSurvivalForestFit):
        raise TypeError("fit must be a RandomSurvivalForestFit")
    if fit.oob is None or fit.inbag_membership is None or fit.training_fingerprint is None:
        raise ValueError("fit must be created with compute_oob=True")
    cell_limit = _integer(max_cells, "max_cells", 1, _MAX_BRIER_CELLS)
    work_limit = _integer(max_work, "max_work", 1, _MAX_OOB_WORK)

    t, e, x = _forest_data(time, event, covariates)
    if x.shape[1] != fit.covariate_count:
        raise ValueError("training covariate count does not match the OOB fit")
    levels = fit.categorical_levels
    if levels and len(levels) != x.shape[1]:
        raise ValueError("fit contains inconsistent categorical level metadata")
    has_categories = any(level is not None for level in levels)
    fingerprint = _forest_fingerprint(t, e, x, levels if has_categories else ())
    if fingerprint != fit.training_fingerprint:
        raise ValueError("training data values or row order do not match the OOB fit")

    oob = fit.oob
    grid = np.asarray(oob.time_grid)
    survival = np.asarray(oob.survival)
    contributors = np.asarray(oob.contributor_count)
    if (
        grid.ndim != 1
        or grid.size == 0
        or grid.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(grid))
        or np.any(grid < 0)
        or np.any(np.diff(grid) <= 0)
        or survival.shape != (t.size, grid.size)
        or survival.dtype != np.dtype(np.float64)
        or contributors.shape != (t.size,)
        or contributors.dtype.kind not in "iu"
        or np.any(contributors < 0)
    ):
        raise ValueError("fit contains inconsistent OOB survival metadata")
    n, m = t.size, grid.size
    live_cells = 2 * n * m + 10 * n + 5 * m
    if live_cells > cell_limit:
        raise ValueError("OOB Brier output and workspace exceed max_cells")
    work = 5 * n * m + n * max(1, ceil(log2(max(2, n)))) + 4 * n + 3 * m
    if work > work_limit:
        raise ValueError("OOB Brier calculation exceeds max_work")

    valid = contributors > 0
    for row_index, has_contributors in enumerate(valid):
        row = survival[row_index]
        if has_contributors:
            if not np.all(np.isfinite(row)) or np.any((row < 0) | (row > 1)):
                raise ValueError("fit contains invalid OOB survival predictions")
        elif np.any(np.isfinite(row)):
            raise ValueError("fit rows without OOB contributors must have missing survival curves")

    censor_survival = _censor_survival(t, e, grid)
    event_grid_count = np.searchsorted(grid, t, side="right")
    event_denominator = np.ones(n, dtype=np.float64)
    has_previous_grid = event_grid_count > 0
    event_denominator[has_previous_grid] = censor_survival[event_grid_count[has_previous_grid] - 1]

    brier = np.full((n, m), np.nan, dtype=np.float64)
    valid_count = int(valid.sum())
    score = np.full(m, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    for j, evaluation_time in enumerate(grid):
        at_risk = valid & (t > evaluation_time)
        failed = valid & (t <= evaluation_time) & (e == 1)
        if np.any(at_risk) and censor_survival[j] <= 0:
            raise ArithmeticError("at-risk IPCW weight is not representable")
        if np.any(failed) and np.any(event_denominator[failed] <= 0):
            raise ArithmeticError("event IPCW weight is not representable")
        values = np.zeros(n, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if np.any(at_risk):
                values[at_risk] = (1.0 - survival[at_risk, j]) ** 2 / censor_survival[j]
            if np.any(failed):
                values[failed] = survival[failed, j] ** 2 / event_denominator[failed]
        selected_values = values[valid_indices]
        if not np.all(np.isfinite(selected_values)):
            raise ArithmeticError("an OOB Brier contribution is not representable")
        brier[valid_indices, j] = selected_values
        if valid_count:
            score[j] = float(np.mean(selected_values))
    crps, crps_standardized = _trapz_over_source_grid(grid, score)
    return RandomSurvivalForestOOBBrier(
        time_grid=_freeze(grid),
        brier=_freeze(brier),
        score=_freeze(score),
        censor_survival=_freeze(censor_survival),
        valid_row_count=valid_count,
        crps=crps,
        crps_standardized=crps_standardized,
    )
