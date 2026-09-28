"""Continuous-covariate incidence surfaces from interval competing-risk fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .interval_competing_risk import (
    IntervalCompetingRiskFit,
    predict_interval_competing_risk,
)


@dataclass(frozen=True)
class IntervalCompetingRiskContour:
    """Both cause-specific incidence surfaces, indexed by covariate then time."""

    fit: IntervalCompetingRiskFit
    continuous_column: int
    cause: int
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    cif1: FloatArray
    cif2: FloatArray

    @property
    def cumulative_incidence(self) -> FloatArray:
        """The selected cause's incidence surface, without copying its values."""
        return self.cif1 if self.cause == 1 else self.cif2


def interval_competing_risk_contour(
    fit: IntervalCompetingRiskFit,
    x: ArrayLike,
    continuous_column: int,
    *,
    cause: int = 1,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
) -> IntervalCompetingRiskContour:
    """Vary one numeric covariate in an already fitted two-cause model.

    Reference data ``x`` determine adjustment means and the default covariate
    grid from its 2.5th to 97.5th percentiles. Default times are 50 points across
    the fitted time range. Joint incidence is checked by the model predictor;
    invalid profiles raise instead of being clipped into a probability range.
    """
    for value, name in ((continuous_column, "continuous_column"), (cause, "cause")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if cause not in (1, 2):
        raise ValueError("cause must be 1 or 2")
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
        if base.shape != (design.shape[1],):
            raise ValueError("profile must contain one value per fitted covariate")
    if grid is None:
        column = design[:, continuous_column] / scales[continuous_column]
        endpoints = np.quantile(column, [0.025, 0.975], method="linear")
        values = np.linspace(endpoints[0], endpoints[1], int(n_grid))
        values *= scales[continuous_column]
    else:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        values = finite(grid, "grid")
    if (
        values.ndim != 1
        or not 2 <= values.size <= 2000
        or not np.isfinite(values).all()
        or np.any(values[1:] <= values[:-1])
    ):
        raise ValueError("grid must contain 2..2000 finite strictly increasing values")
    if times is None:
        prediction_times = np.linspace(fit.boundary_knots[0], fit.boundary_knots[1], 50)
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        prediction_times = finite(times, "times")
    if (
        prediction_times.ndim != 1
        or not 1 <= prediction_times.size <= 100_000
        or np.any(prediction_times[1:] <= prediction_times[:-1])
        or np.any(prediction_times < fit.boundary_knots[0])
        or np.any(prediction_times > fit.boundary_knots[1])
    ):
        raise ValueError("times must be strictly increasing within the fitted time range")
    if not np.isfinite(base).all():
        raise ArithmeticError("contour adjustment profile is not representable")
    cells = (
        12 * values.size * prediction_times.size
        + values.size * design.shape[1]
        + 2 * prediction_times.size * (fit.knots.size + 4)
    )
    if cells > 2_000_000:
        raise ValueError("combined interval competing-risk contour exceeds 2,000,000 cells")
    profiles = np.repeat(base[None, :], values.size, axis=0)
    profiles[:, continuous_column] = values
    prediction = predict_interval_competing_risk(fit, prediction_times, profiles)
    return IntervalCompetingRiskContour(
        fit,
        int(continuous_column),
        int(cause),
        _freeze(base),
        _freeze(values),
        prediction.times,
        prediction.cif1,
        prediction.cif2,
    )


def plot_interval_competing_risk_contour_2d(
    result: IntervalCompetingRiskContour, *, ax: Any | None = None, levels: int = 12
) -> Any:
    """Plot the selected cause's cumulative incidence; no confidence band."""
    if result.times.size < 2:
        raise ValueError("contour plotting requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    contours = ax.contourf(result.grid, result.times, result.cumulative_incidence.T, levels=levels)
    ax.figure.colorbar(contours, ax=ax, label=f"Cause {result.cause} cumulative incidence")
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    return ax


def plot_interval_competing_risk_contour_3d(
    result: IntervalCompetingRiskContour, *, ax: Any | None = None
) -> Any:
    """Plot the selected cause's cumulative-incidence surface."""
    if result.times.size < 2:
        raise ValueError("surface plotting requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    xx, tt = np.meshgrid(result.grid, result.times)
    ax.plot_surface(xx, tt, result.cumulative_incidence.T)
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    ax.set_zlabel(f"Cause {result.cause} cumulative incidence")
    return ax
