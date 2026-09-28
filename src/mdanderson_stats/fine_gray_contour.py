"""Fine–Gray cumulative-incidence grids following SurvivalContour's contract."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .fine_gray import FineGrayFit, fine_gray, fine_gray_predict


@dataclass(frozen=True)
class FineGrayContour:
    """Target-cause incidence over covariate values (rows) and times (columns)."""

    fit: FineGrayFit
    continuous_column: int
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    cumulative_incidence: FloatArray
    cumulative_subdistribution_hazard: FloatArray


def fine_gray_contour(
    time: ArrayLike,
    status: ArrayLike,
    x: ArrayLike,
    continuous_column: int,
    *,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    censoring_groups: ArrayLike | None = None,
    failcode: int = 1,
    cencode: int = 0,
) -> FineGrayContour:
    """Fit fixed-effect Fine–Gray regression and construct an incidence contour.

    The default grid spans empirical 2.5th..97.5th covariate percentiles; other
    columns use training means or an explicit complete profile. Default times
    are target-event times plus zero. Explicit times use right-continuous steps.
    Time interactions are available through fine_gray/fine_gray_predict; this
    convenience builder varies one column of a fixed numeric design.
    """
    if isinstance(continuous_column, (bool, np.bool_)) or not isinstance(
        continuous_column, (int, np.integer)
    ):
        raise ValueError("continuous_column must be an integer column index")
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)):
        raise ValueError("n_grid must be an integer in 2..2000")
    if not 2 <= n_grid <= 2000:
        raise ValueError("n_grid must be in 2..2000")
    if any(np.iscomplexobj(a) for a in (time, status, x)):
        raise ValueError("time, status and x must be real")
    t, code, design = finite(time, "time"), count(status, "status"), finite(x, "x")
    if design.ndim == 1:
        design = design[:, None]
    if (
        design.ndim != 2
        or not 0 <= continuous_column < design.shape[1]
        or t.ndim != 1
        or code.shape != t.shape
        or design.shape[0] != t.size
        or not 2 <= t.size <= 100_000
        or design.shape[1] > 100
        or design.size > 2_000_000
        or np.any(t < 0)
    ):
        raise ValueError("require matching nonnegative time/status/x and a valid continuous column")
    for value, name in ((failcode, "failcode"), (cencode, "cencode")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= value < 2**53:
            raise ValueError(f"{name} must be a nonnegative integer smaller than 2**53")
    if failcode == cencode or not np.any(code == failcode):
        raise ValueError("distinct failcode/cencode and at least one target event are required")
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1
    normalized = design / scale
    base_profile = np.mean(normalized, axis=0) * scale
    if profile is not None:
        if np.iscomplexobj(profile):
            raise ValueError("profile must be real")
        base_profile = finite(profile, "profile")
        if base_profile.shape != (design.shape[1],):
            raise ValueError("profile must contain one value per covariate")
    if grid is None:
        low, high = np.quantile(normalized[:, continuous_column], [0.025, 0.975])
        values = np.linspace(low, high, n_grid) * scale[continuous_column]
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
        prediction_times = np.unique(np.r_[0.0, t[code == failcode]])
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
        raise ValueError("times must contain 1..100000 increasing nonnegative values")
    # Preflight before fitting: include predictor output/work arrays and profiles.
    cells = 4 * values.size * prediction_times.size + values.size * design.shape[1]
    if cells > 2_000_000:
        raise ValueError("combined Fine–Gray contour output exceeds 2,000,000 cells")
    fit = fine_gray(
        t, code, design, censoring_groups=censoring_groups, failcode=failcode, cencode=cencode
    )
    profiles = np.repeat(base_profile[None, :], values.size, axis=0)
    profiles[:, continuous_column] = values
    prediction = fine_gray_predict(fit, profiles, times=prediction_times)
    return FineGrayContour(
        fit,
        int(continuous_column),
        _freeze(base_profile),
        _freeze(values),
        prediction.times,
        prediction.cumulative_incidence,
        prediction.cumulative_subdistribution_hazard,
    )


def plot_fine_gray_contour_2d(
    result: FineGrayContour, *, ax: Any | None = None, levels: int = 12
) -> Any:
    """Draw target-cause incidence; requires the optional Matplotlib extra."""
    if result.times.size < 2:
        raise ValueError("a contour plot requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    contours = ax.contourf(result.grid, result.times, result.cumulative_incidence.T, levels=levels)
    ax.figure.colorbar(contours, ax=ax, label="Cumulative incidence")
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    return ax


def plot_fine_gray_contour_3d(result: FineGrayContour, *, ax: Any | None = None) -> Any:
    """Draw the target-cause incidence surface without confidence bands."""
    if result.times.size < 2:
        raise ValueError("a contour plot requires at least two prediction times")
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    xx, tt = np.meshgrid(result.grid, result.times)
    ax.plot_surface(xx, tt, result.cumulative_incidence.T)
    ax.set_xlabel(f"Covariate column {result.continuous_column}")
    ax.set_ylabel("Time")
    ax.set_zlabel("Cumulative incidence")
    return ax
