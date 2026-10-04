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
    _freeze_index,
    _integer,
    fit_random_survival_forest,
    predict_random_survival_forest,
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
    score_row_count: np.ndarray | None = None
    censor_model: str = "km"
    censor_random_state: int | None = None
    censor_forest_fit: RandomSurvivalForestFit | None = None


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


def _source_nodesize(n: int, p: int) -> int:
    """Match the Brier helper's ``set.nodesize`` defaults."""
    if n <= 300:
        return 2 if p > n else 5
    if n <= 2000:
        return 10
    return n // 200


def _project_censor_survival(
    censor_grid: FloatArray, censor_by_row: FloatArray, outcome_grid: FloatArray
) -> FloatArray:
    """Project each censor curve by the source ``sum(censor_grid <= t)`` rule."""
    if (
        censor_grid.ndim != 1
        or censor_grid.size == 0
        or outcome_grid.ndim != 1
        or censor_by_row.ndim != 2
        or censor_by_row.shape[1] != censor_grid.size
        or np.any(~np.isfinite(censor_grid))
        or np.any(np.diff(censor_grid) <= 0)
        or np.any(~np.isfinite(outcome_grid))
        or np.any(~np.isfinite(censor_by_row))
        or np.any((censor_by_row < 0) | (censor_by_row > 1))
    ):
        raise ArithmeticError("censor forest returned invalid survival curves")
    indices = np.searchsorted(censor_grid, outcome_grid, side="right")
    projected = np.ones((censor_by_row.shape[0], outcome_grid.size), dtype=np.float64)
    has_previous = indices > 0
    projected[:, has_previous] = censor_by_row[:, indices[has_previous] - 1]
    return projected


def _rfsrc_censor_survival(
    time: FloatArray,
    event: FloatArray,
    covariates: FloatArray,
    grid: FloatArray,
    fit: RandomSurvivalForestFit,
    seed: int,
    max_cells: int,
    max_work: int,
    base_work: int,
) -> tuple[FloatArray, RandomSurvivalForestFit | None, int | None]:
    """Fit the source-configured censoring forest and project row curves."""
    n, p, m = time.size, covariates.shape[1], grid.size
    if not np.any(event == 0):
        # The native helper skips fitting here, although its later matrix
        # indexing errors because this branch returns a vector rather than a
        # subject-by-time matrix. G(t)=1 is the direct no-censoring contract.
        if 4 * n * m + 10 * n + 5 * m > max_cells:
            raise ValueError("censor and Brier outputs/workspace exceed max_cells")
        return np.ones((n, m), dtype=np.float64), None, None
    if p == 0:
        raise ValueError("censor_model='rfsrc' requires at least one covariate")

    nodesize = _source_nodesize(n, p)
    sample_size = int(np.rint(0.632 * n))
    max_leaves_per_tree = max(1, sample_size)
    theoretical_nodes = 50 * (2 * max_leaves_per_tree - 1)
    # Bound the Brier matrices, the censor predictions/projection, packed tree
    # nodes and leaf-step records before the secondary fit consumes RNG.
    predictor_grid_upper = min(150, int(np.unique(time[event == 0]).size))
    max_factor_levels = max(
        (int(levels.size) // 2 for levels in fit.categorical_levels if levels is not None),
        default=0,
    )
    has_categorical_columns = any(levels is not None for levels in fit.categorical_levels)
    categorical_profile_copy = n * p if has_categorical_columns else 0
    brier_cells = 4 * n * m + 10 * n + 5 * m
    prediction_cells = (
        8 * n * predictor_grid_upper + 2 * n * p + predictor_grid_upper + categorical_profile_copy
    )
    leaf_cells = 10 * 50 * sample_size
    node_cells_per_node = 64 + max_factor_levels
    fixed_cells = brier_cells + prediction_cells + n * m + leaf_cells
    max_nodes_for_fit = min(
        theoretical_nodes, max(0, max_cells - fixed_cells) // node_cells_per_node
    )
    if max_nodes_for_fit < 50:
        raise ValueError("censor forest, Brier outputs, and workspace exceed max_cells")
    minimum_fit_work = 50 * sample_size
    minimum_prediction_work = 50 * n * predictor_grid_upper
    if base_work + minimum_fit_work + minimum_prediction_work > max_work:
        raise ValueError("censor forest fit/prediction exceeds max_work")

    categorical_features = tuple(
        i for i, levels in enumerate(fit.categorical_levels) if levels is not None
    )
    censor_fit = fit_random_survival_forest(
        time,
        1.0 - event,
        covariates,
        categorical_features=categorical_features,
        split_rule="random",
        n_trees=50,
        nodesize=nodesize,
        nsplit=1,
        random_state=seed,
        compute_oob=False,
        max_nodes=max(50, max_nodes_for_fit),
        max_leaf_event_records=max(1, 50 * sample_size),
        max_sampled_rows=max(1, 50 * sample_size),
        max_split_work=max(1, max_work - base_work - minimum_prediction_work - minimum_fit_work),
    )
    remaining_prediction_work = max_work - base_work - minimum_fit_work - censor_fit.split_work
    if remaining_prediction_work < 1:
        raise ValueError("censor forest fit exhausted the shared max_work budget")
    censor_prediction = predict_random_survival_forest(
        censor_fit,
        profiles=covariates,
        max_output_cells=max_cells,
        max_prediction_work=remaining_prediction_work,
    )
    censor_grid = np.asarray(censor_prediction.times)
    censor_by_row = np.asarray(censor_prediction.survival)
    if censor_by_row.shape[0] != n:
        raise ArithmeticError("censor forest returned one curve per unexpected row")
    projected = _project_censor_survival(censor_grid, censor_by_row, grid)
    return projected, censor_fit, seed


def _rfsrc_brier_components(
    time: FloatArray,
    event: FloatArray,
    grid: FloatArray,
    survival: FloatArray,
    censor_survival: FloatArray,
    valid: np.ndarray,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Apply both literal source IPCW terms to row-specific censor curves."""
    n, m = time.size, grid.size
    if censor_survival.shape != (n, m):
        raise ValueError("row-specific censor survival must match training rows and event grid")
    event_grid_count = np.searchsorted(grid, time, side="right")
    event_denominator = np.ones(n, dtype=np.float64)
    has_previous_grid = event_grid_count > 0
    rows = np.arange(n)
    event_denominator[has_previous_grid] = censor_survival[
        rows[has_previous_grid], event_grid_count[has_previous_grid] - 1
    ]
    brier = np.full((n, m), np.nan, dtype=np.float64)
    score = np.full(m, np.nan, dtype=np.float64)
    score_row_count = np.zeros(m, dtype=np.int64)
    for j, evaluation_time in enumerate(grid):
        at_risk = time > evaluation_time
        failed = (time <= evaluation_time) & (event == 1)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            c1 = (failed * 1.0) / event_denominator
            c2 = (at_risk * 1.0) / censor_survival[:, j]
            weights = c1 + c2
            contributions = (1.0 * at_risk - survival[:, j]) ** 2 * weights
        if np.any(np.isinf(contributions[valid])):
            raise ArithmeticError("an OOB IPCW contribution is not representable")
        contributes = valid & ~np.isnan(contributions)
        brier[contributes, j] = contributions[contributes]
        score_row_count[j] = int(np.count_nonzero(contributes))
        if score_row_count[j]:
            score[j] = float(np.mean(contributions[contributes]))
    return brier, score, score_row_count


def random_survival_forest_oob_brier_score(
    fit: RandomSurvivalForestFit,
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    censor_model: str = "km",
    censor_random_state: int = 0,
    max_cells: int = _MAX_BRIER_CELLS,
    max_work: int = _MAX_OOB_WORK,
) -> RandomSurvivalForestOOBBrier:
    """Compute the pinned RF-SRC OOB IPCW Brier score and integrated CRPS.

    This API deliberately supports the complete training population only:
    native subset handling in the cached helper is ambiguous. The censoring
    distribution uses all supplied training outcomes, while rows with no OOB
    forest prediction are omitted from per-time score averages. The default
    ``censor_model="km"`` preserves the source's global Nelson--Aalen
    censoring estimate. ``censor_model="rfsrc"`` fits a separate source-sized
    random-split forest and accepts an explicit reproducible seed.
    """
    if not isinstance(fit, RandomSurvivalForestFit):
        raise TypeError("fit must be a RandomSurvivalForestFit")
    if fit.oob is None or fit.inbag_membership is None or fit.training_fingerprint is None:
        raise ValueError("fit must be created with compute_oob=True")
    cell_limit = _integer(max_cells, "max_cells", 1, _MAX_BRIER_CELLS)
    work_limit = _integer(max_work, "max_work", 1, _MAX_OOB_WORK)
    if not isinstance(censor_model, str) or censor_model not in ("km", "rfsrc"):
        raise ValueError("censor_model must be 'km' or 'rfsrc'")
    censor_seed = (
        _integer(censor_random_state, "censor_random_state", 0, np.iinfo(np.int32).max)
        if censor_model == "rfsrc"
        else None
    )

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

    censor_forest_fit: RandomSurvivalForestFit | None = None
    if censor_model == "km":
        censor_survival = _censor_survival(t, e, grid)
    else:
        assert censor_seed is not None
        censor_survival, censor_forest_fit, _ = _rfsrc_censor_survival(
            t, e, x, grid, fit, censor_seed, cell_limit, work_limit, work
        )
    if censor_model == "rfsrc":
        brier, score, score_row_count = _rfsrc_brier_components(
            t, e, grid, survival, censor_survival, valid
        )
        crps, crps_standardized = _trapz_over_source_grid(grid, score)
        return RandomSurvivalForestOOBBrier(
            time_grid=_freeze(grid),
            brier=_freeze(brier),
            score=_freeze(score),
            censor_survival=_freeze(censor_survival),
            valid_row_count=int(valid.sum()),
            crps=crps,
            crps_standardized=crps_standardized,
            score_row_count=_freeze_index(score_row_count, np.dtype(np.int64)),
            censor_model=censor_model,
            censor_random_state=censor_seed,
            censor_forest_fit=censor_forest_fit,
        )

    event_grid_count = np.searchsorted(grid, t, side="right")
    event_denominator = np.ones(n, dtype=np.float64)
    has_previous_grid = event_grid_count > 0
    event_denominator[has_previous_grid] = censor_survival[event_grid_count[has_previous_grid] - 1]
    brier = np.full((n, m), np.nan, dtype=np.float64)
    valid_count = int(valid.sum())
    score = np.full(m, np.nan, dtype=np.float64)
    score_row_count = np.zeros(m, dtype=np.int64)
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
            score_row_count[j] = valid_count
    crps, crps_standardized = _trapz_over_source_grid(grid, score)
    return RandomSurvivalForestOOBBrier(
        time_grid=_freeze(grid),
        brier=_freeze(brier),
        score=_freeze(score),
        censor_survival=_freeze(censor_survival),
        valid_row_count=valid_count,
        crps=crps,
        crps_standardized=crps_standardized,
        score_row_count=_freeze_index(score_row_count, np.dtype(np.int64)),
        censor_model=censor_model,
        censor_random_state=censor_seed,
        censor_forest_fit=censor_forest_fit,
    )
