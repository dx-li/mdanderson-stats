"""Continuous-covariate survival surfaces from an already fitted forest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .random_survival_forest import RandomSurvivalForestFit, predict_random_survival_forest


@dataclass(frozen=True)
class RandomSurvivalForestContour:
    """Forest survival and Nelson–Aalen surfaces; no confidence bounds."""

    fit: RandomSurvivalForestFit
    continuous_column: int
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    quantile_probabilities: FloatArray
    covariate_quantiles: FloatArray
    quantile_profiles: FloatArray
    quantile_survival: FloatArray
    quantile_cumulative_hazard: FloatArray


def random_survival_forest_contour(
    fit: RandomSurvivalForestFit,
    x: ArrayLike,
    continuous_column: int,
    *,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    quantile_probabilities: ArrayLike = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> RandomSurvivalForestContour:
    """Vary one covariate in a fitted forest without growing another forest.

    ``x`` provides the numeric reference population for adjustment means and
    empirical covariate quantiles. Its columns must match the fitted model.
    Default times are the fitted event grid with zero prepended when absent.
    Curves are right-continuous; cumulative hazard is the mean leaf
    Nelson–Aalen estimate, not minus log ensemble survival.
    """
    if isinstance(continuous_column, (bool, np.bool_)) or not isinstance(
        continuous_column, (int, np.integer)
    ):
        raise ValueError("continuous_column must be an integer column index")
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)):
        raise ValueError("n_grid must be an integer in 2..2000")
    if not 2 <= n_grid <= 2000:
        raise ValueError("n_grid must be in 2..2000")
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    design = finite(x, "x")
    if design.ndim == 1:
        design = design[:, None]
    if (
        design.ndim != 2
        or design.shape[1] != fit.covariate_mean.size
        or not 1 <= design.shape[0] <= 100_000
        or design.size > 2_000_000
        or not 0 <= continuous_column < design.shape[1]
    ):
        raise ValueError("x must match fitted columns and contain at most 2,000,000 values")
    scales = np.max(np.abs(design), axis=0)
    scales[scales == 0] = 1.0
    if profile is None:
        base = np.mean(design / scales, axis=0) * scales
    else:
        if np.iscomplexobj(profile):
            raise ValueError("profile must be real")
        base = finite(profile, "profile")
        if base.ndim != 1 or base.size != design.shape[1]:
            raise ValueError("profile must contain one value per fitted covariate")
    normalized_column = design[:, continuous_column] / scales[continuous_column]
    if grid is None:
        endpoints = np.quantile(normalized_column, [0.025, 0.975], method="linear")
        if endpoints[1] <= endpoints[0]:
            raise ValueError("continuous covariate must vary to form a grid")
        values = np.linspace(float(endpoints[0]), float(endpoints[1]), int(n_grid))
        values *= scales[continuous_column]
    else:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        values = finite(grid, "grid")
        if values.ndim != 1 or not 2 <= values.size <= 2000 or np.any(values[1:] <= values[:-1]):
            raise ValueError("grid must be strictly increasing with 2..2000 values")
    if times is None:
        prediction_times = fit.time_grid
        if prediction_times[0] > 0:
            prediction_times = np.r_[0.0, prediction_times]
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        prediction_times = finite(times, "times")
        if (
            prediction_times.ndim != 1
            or not 1 <= prediction_times.size <= 100_000
            or np.any(prediction_times < 0)
            or np.any(prediction_times[1:] <= prediction_times[:-1])
        ):
            raise ValueError("times must be a strictly increasing nonnegative vector")
    if np.iscomplexobj(quantile_probabilities):
        raise ValueError("quantile_probabilities must be real")
    probabilities = finite(quantile_probabilities, "quantile_probabilities")
    if (
        probabilities.ndim != 1
        or not 1 <= probabilities.size <= 20
        or np.any((probabilities <= 0) | (probabilities >= 1))
        or np.unique(probabilities).size != probabilities.size
    ):
        raise ValueError("quantile_probabilities must contain 1..20 distinct values in (0,1)")
    cells = (
        8 * (values.size + probabilities.size) * prediction_times.size
        + (values.size + probabilities.size) * design.shape[1]
    )
    if cells > 2_000_000:
        raise ValueError("combined forest contour output exceeds 2,000,000 cells")
    covariate_quantiles = (
        np.quantile(normalized_column, probabilities, method="linear") * scales[continuous_column]
    )
    if not all(np.isfinite(a).all() for a in (base, values, covariate_quantiles)):
        raise ArithmeticError("contour profile or covariate grid is not representable")
    main_profiles = np.repeat(base[None, :], values.size, axis=0)
    main_profiles[:, continuous_column] = values
    quantile_profiles = np.repeat(base[None, :], probabilities.size, axis=0)
    quantile_profiles[:, continuous_column] = covariate_quantiles
    main = predict_random_survival_forest(fit, prediction_times, main_profiles)
    quantile = predict_random_survival_forest(fit, prediction_times, quantile_profiles)
    return RandomSurvivalForestContour(
        fit,
        int(continuous_column),
        _freeze(base),
        _freeze(values),
        _freeze(prediction_times),
        main.survival,
        main.cumulative_hazard,
        _freeze(probabilities),
        _freeze(covariate_quantiles),
        _freeze(quantile_profiles),
        quantile.survival,
        quantile.cumulative_hazard,
    )
