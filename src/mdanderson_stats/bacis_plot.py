"""Optional plotting for the analytical BaCIS classification posterior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .bacis import BaCISClassification
from .bacis_theta import _precision, bacis_theta_posterior

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_NATIVE_WINDOW = (-100.0, 100.0)
_NATIVE_COLORS = ("brown", "red", "orange", "blue", "green")
_MAX_POINTS = 10_001
_MAX_CELLS = 200_000


def _window(
    value: tuple[float, float] | Literal["native", "auto"] | None, scale: float
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "native":
            return _NATIVE_WINDOW
        if value == "auto":
            extent = 4.0 * scale
            return (-extent, extent)
        raise ValueError("xlim must be 'native', 'auto', None or a finite increasing pair")
    if np.iscomplexobj(value):
        raise ValueError("xlim must be real")
    limits = np.asarray(value)
    if limits.shape != (2,) or limits.dtype.kind not in "iuf":
        raise ValueError("xlim must be a finite increasing pair")
    result = np.asarray(limits, dtype=np.float64)
    if not np.isfinite(result).all() or result[0] >= result[1]:
        raise ValueError("xlim must be a finite increasing pair")
    return (float(result[0]), float(result[1]))


def _plot_grid(
    grid: ArrayLike | None,
    window: tuple[float, float],
    precision: float,
    points: int,
    groups: int,
) -> FloatArray:
    if grid is not None:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        values = np.asarray(grid)
        if (
            values.ndim != 1
            or not 2 <= values.size <= _MAX_POINTS
            or values.dtype.kind not in "iuf"
        ):
            raise ValueError(f"grid must be a real vector with 2..{_MAX_POINTS} values")
        theta = np.asarray(values, dtype=np.float64)
        if not np.isfinite(theta).all() or np.any(theta[1:] <= theta[:-1]):
            raise ValueError("grid must be finite and strictly increasing")
    else:
        base = np.linspace(*window, points)
        # Keep the exact density jump and resolve narrow peaks when precision is high.
        sd = 1.0 / np.sqrt(precision)
        central_low = max(window[0], -5.0 * sd)
        central_high = min(window[1], 5.0 * sd)
        if central_low < central_high:
            central = np.linspace(central_low, central_high, 201)
            theta = np.unique(np.r_[base, central])
        else:
            theta = base
    if theta.size * groups > _MAX_CELLS:
        raise ValueError("grid * number of groups must be at most 200,000")
    return np.asarray(theta, dtype=np.float64)


def plot_bacis_classification_posterior(
    classification: BaCISClassification,
    *,
    latent_precision: float = 0.001,
    grid: ArrayLike | None = None,
    xlim: tuple[float, float] | Literal["native", "auto"] | None = None,
    points: int = 1001,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
) -> Axes:
    """Plot exact piecewise posterior densities for BaCIS subgroup effects.

    The native CRAN routine shows sampled-chain KDEs on ``[-100,100]``. This
    Python helper evaluates the analytical posterior instead. Each group's
    negative and nonnegative branches are drawn separately, including their
    exact one-sided limits at zero, so the density jump is not connected by a
    false line. With no supplied axes, ``xlim=None`` uses the native window;
    ``xlim="auto"`` uses four prior standard deviations. With supplied axes,
    ``xlim=None`` preserves their existing view. An explicit pair, ``"native"``
    or ``"auto"`` overrides it. New axes autoscale the density vertically;
    supplied axes retain their labels and limits.
    """
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise ValueError("points must be an integer")
    if not 101 <= points <= _MAX_POINTS:
        raise ValueError(f"points must be in [101, {_MAX_POINTS}]")
    if not isinstance(classification, BaCISClassification):
        raise TypeError("classification must be a BaCISClassification")
    precision = _precision(latent_precision)
    sd = 1.0 / np.sqrt(precision)
    requested_window = _window(xlim, sd)

    if ax is not None and requested_window is None:
        view = ax.get_xlim()
        window = (min(float(view[0]), float(view[1])), max(float(view[0]), float(view[1])))
    else:
        window = _NATIVE_WINDOW if requested_window is None else requested_window

    low_probability = np.asarray(classification.low_probability, dtype=np.float64)
    groups = int(low_probability.size)
    if not 1 <= groups <= 100:
        raise ValueError("classification must contain 1..100 groups")
    theta = _plot_grid(grid, window, precision, int(points), groups)
    posterior = bacis_theta_posterior(classification, theta, latent_precision=precision)

    if labels is None:
        group_labels = tuple(f"Arm {index + 1}" for index in range(groups))
    else:
        if (
            isinstance(labels, str)
            or len(labels) != groups
            or any(not isinstance(s, str) for s in labels)
        ):
            raise ValueError("labels must contain one string per group")
        group_labels = tuple(labels)
    if colors is None:
        group_colors = tuple(_NATIVE_COLORS[index % len(_NATIVE_COLORS)] for index in range(groups))
    else:
        if (
            isinstance(colors, str)
            or len(colors) != groups
            or any(not isinstance(c, str) for c in colors)
        ):
            raise ValueError("colors must contain one color string per group")
        group_colors = tuple(colors)

    created_axes = ax is None
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    supplied_xlim = ax.get_xlim() if not created_axes else None
    supplied_ylim = ax.get_ylim() if not created_axes else None
    supplied_labels = (
        (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) if not created_axes else None
    )

    left_of_zero = theta < 0
    right_of_zero = theta > 0
    includes_left_limit = theta[0] < 0 and theta[-1] >= 0
    includes_right_limit = theta[0] <= 0 and theta[-1] > 0
    zero_density_scale = np.sqrt(precision) * np.sqrt(2.0 / np.pi)
    maximum_density = 0.0
    for group in range(groups):
        negative_theta = theta[left_of_zero]
        negative_density = posterior.density[left_of_zero, group]
        positive_theta = theta[right_of_zero]
        positive_density = posterior.density[right_of_zero, group]
        if includes_left_limit:
            negative_theta = np.r_[negative_theta, 0.0]
            negative_density = np.r_[
                negative_density,
                zero_density_scale * classification.low_probability[group],
            ]
        if includes_right_limit:
            positive_theta = np.r_[0.0, positive_theta]
            positive_density = np.r_[
                zero_density_scale * classification.high_probability[group],
                positive_density,
            ]
        if negative_theta.size:
            ax.plot(
                negative_theta,
                negative_density,
                color=group_colors[group],
                linewidth=2.0,
                label=group_labels[group] if not positive_theta.size else None,
            )
        if positive_theta.size:
            ax.plot(
                positive_theta,
                positive_density,
                color=group_colors[group],
                linewidth=2.0,
                label=group_labels[group],
            )
        if negative_density.size:
            maximum_density = max(maximum_density, float(np.max(negative_density)))
        if positive_density.size:
            maximum_density = max(maximum_density, float(np.max(positive_density)))

    if requested_window is not None or created_axes:
        ax.set_xlim(window)
    if created_axes:
        upper = maximum_density * 1.05 if maximum_density > 0 else 1.0
        ax.set_ylim(0.0, upper)
        ax.set_title("BaCIS classification posterior for theta")
        ax.set_xlabel("theta")
        ax.set_ylabel("Posterior density")
    else:
        if requested_window is None and supplied_xlim is not None:
            ax.set_xlim(supplied_xlim)
        if supplied_ylim is not None:
            ax.set_ylim(supplied_ylim)
        if supplied_labels is not None:
            ax.set_xlabel(supplied_labels[0])
            ax.set_ylabel(supplied_labels[1])
            ax.set_title(supplied_labels[2])
    ax.legend(loc="best", fontsize=8)
    return ax
