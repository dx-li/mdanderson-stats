"""Continuous-covariate identification bounds from interval-censored PH fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .interval_survival import IntervalSurvivalFit, predict_interval_survival


@dataclass(frozen=True)
class IntervalSurvivalContour:
    """Lower and upper identified survival surfaces, not confidence limits."""

    fit: IntervalSurvivalFit
    continuous_column: int
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    survival_lower: FloatArray
    survival_upper: FloatArray
    quantile_probabilities: FloatArray
    covariate_quantiles: FloatArray
    quantile_profiles: FloatArray
    quantile_survival_lower: FloatArray
    quantile_survival_upper: FloatArray


def interval_survival_contour(
    fit: IntervalSurvivalFit,
    x: ArrayLike,
    continuous_column: int,
    *,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    quantile_probabilities: ArrayLike = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> IntervalSurvivalContour:
    """Vary one covariate in a fitted interval-censored PH model.

    ``x`` supplies numeric adjustment means and empirical covariate quantiles.
    Default times are zero and all finite support endpoints. Both returned
    surfaces preserve the baseline distribution's unidentified location within
    each support interval; they are not statistical confidence bounds.
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
        endpoints = np.r_[0.0, fit.support_lower, fit.support_upper]
        prediction_times = np.unique(endpoints[np.isfinite(endpoints)])
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
        12 * (values.size + probabilities.size) * prediction_times.size
        + (values.size + probabilities.size) * design.shape[1]
    )
    if cells > 2_000_000:
        raise ValueError("combined interval contour output exceeds 2,000,000 cells")
    covariate_quantiles = (
        np.quantile(normalized_column, probabilities, method="linear") * scales[continuous_column]
    )
    if not all(np.isfinite(a).all() for a in (base, values, covariate_quantiles)):
        raise ArithmeticError("contour profile or covariate grid is not representable")
    main_profiles = np.repeat(base[None, :], values.size, axis=0)
    main_profiles[:, continuous_column] = values
    quantile_profiles = np.repeat(base[None, :], probabilities.size, axis=0)
    quantile_profiles[:, continuous_column] = covariate_quantiles
    main = predict_interval_survival(fit, prediction_times, main_profiles)
    quantile = predict_interval_survival(fit, prediction_times, quantile_profiles)
    return IntervalSurvivalContour(
        fit,
        int(continuous_column),
        _freeze(base),
        _freeze(values),
        _freeze(prediction_times),
        main.survival_lower,
        main.survival_upper,
        _freeze(probabilities),
        _freeze(covariate_quantiles),
        _freeze(quantile_profiles),
        quantile.survival_lower,
        quantile.survival_upper,
    )


def plot_interval_survival_contour_2d(
    result: IntervalSurvivalContour,
    *,
    bound: str,
    ax: Any | None = None,
    levels: int = 12,
) -> Any:
    """Plot an explicitly selected identification bound, not a confidence band."""
    if bound not in ("lower", "upper"):
        raise ValueError("bound must be lower or upper")
    if result.times.size < 2:
        raise ValueError("contour plotting requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    values = result.survival_lower if bound == "lower" else result.survival_upper
    contours = ax.contourf(result.grid, result.times, values.T, levels=levels)
    ax.figure.colorbar(contours, ax=ax, label=f"{bound.title()} identified survival bound")
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    return ax


def plot_interval_survival_contour_3d(
    result: IntervalSurvivalContour,
    *,
    bound: str,
    ax: Any | None = None,
) -> Any:
    """Plot one of the fitted model's two survival identification surfaces."""
    if bound not in ("lower", "upper"):
        raise ValueError("bound must be lower or upper")
    if result.times.size < 2:
        raise ValueError("surface plotting requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    values = result.survival_lower if bound == "lower" else result.survival_upper
    xx, tt = np.meshgrid(result.grid, result.times)
    ax.plot_surface(xx, tt, values.T)
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    ax.set_zlabel(f"{bound.title()} survival bound")
    return ax
