"""Optional plots for median-effect and Loewe interaction-index calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from .interaction_index import InteractionIndex
from .median_effect import MedianEffectFit

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_MAX_POINTS = 10_000


def _vector(value: ArrayLike, name: str) -> np.ndarray:
    """Validate bounded vector input before converting Python sequences."""
    raw: object
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= _MAX_POINTS:
            raise ValueError(f"{name} must be a vector with 1..{_MAX_POINTS} values")
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_POINTS:
            raise ValueError(f"{name} must be a vector with 1..{_MAX_POINTS} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain scalar values")
        raw = value
    else:
        raise ValueError(f"{name} must be a one-dimensional array, list, or tuple")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain finite real values")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    from matplotlib import pyplot as plt

    _, created = plt.subplots()
    return created


def plot_median_effect(
    fit: MedianEffectFit,
    dose: ArrayLike,
    effect: ArrayLike,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot observed log-dose/logit-effect pairs and an existing fitted line.

    The fit is not recalculated. Effects must lie strictly inside (0, 1), as
    required by :func:`fit_median_effect`; at most 10,000 observations are
    accepted to keep this presentation helper bounded.
    """
    if not isinstance(fit, MedianEffectFit):
        raise TypeError("fit must be a MedianEffectFit")
    d, y = _vector(dose, "dose"), _vector(effect, "effect")
    if d.shape != y.shape or np.any(d <= 0) or np.any((y <= 0) | (y >= 1)):
        raise ValueError("dose and effect must match; doses >0 and effects in (0,1) are required")
    x = np.log(d)
    z = np.log(y) - np.log1p(-y)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(z)):
        raise ValueError("log-dose and logit-effect coordinates must be finite")
    line_x = np.array([np.min(x), np.max(x)])
    with np.errstate(over="ignore", invalid="ignore"):
        line_z = fit.intercept + fit.slope * line_x
    if np.any(~np.isfinite(line_z)):
        raise ArithmeticError("fitted line is not representable on the observed log-dose range")

    axes = _axes(ax)
    axes.scatter(x, z, label=None if label is None else f"{label} observations")
    axes.plot(line_x, line_z, label=None if label is None else f"{label} fit")
    axes.set(xlabel="log(dose)", ylabel="logit(effect)")
    return axes


def plot_interaction_index(
    effect: ArrayLike,
    result: InteractionIndex,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    kind: Literal["line", "points"] = "line",
    log_index: bool = False,
) -> Axes:
    """Plot pointwise interaction-index estimates and intervals.

    ``kind="line"`` sorts by effect and connects ordered values; duplicate
    effects are rejected in this mode. ``kind="points"`` preserves input order.
    The intervals are pointwise delta-method intervals, not simultaneous bands.
    Set ``log_index=True`` to plot log index without exponentiating (useful when
    the raw index is outside floating-point range). Additivity is 1 in raw
    coordinates and 0 in log coordinates.
    """
    if not isinstance(result, InteractionIndex):
        raise TypeError("result must be an InteractionIndex")
    if kind not in ("line", "points"):
        raise ValueError("kind must be 'line' or 'points'")
    x = _vector(effect, "effect")
    raw_log = result.log_index
    raw_interval = result.log_interval
    if (
        not isinstance(raw_log, np.ndarray)
        or raw_log.ndim != 1
        or raw_log.size != x.size
        or raw_log.size > _MAX_POINTS
        or not isinstance(raw_interval, np.ndarray)
        or raw_interval.shape != (x.size, 2)
        or np.iscomplexobj(raw_log)
        or np.iscomplexobj(raw_interval)
    ):
        raise ValueError("effect and interaction result must have matching bounded vectors")
    estimate = np.asarray(raw_log, dtype=float)
    interval = np.asarray(raw_interval, dtype=float)
    if (
        np.any((x <= 0) | (x >= 1))
        or np.any(~np.isfinite(estimate))
        or np.any(~np.isfinite(interval))
        or np.any(interval[:, 0] > estimate)
        or np.any(interval[:, 1] < estimate)
    ):
        raise ValueError("effects must be in (0,1) and finite intervals must contain estimates")
    if kind == "line" and np.unique(x).size != x.size:
        raise ValueError("line plots require unique effect coordinates")

    if not log_index:
        with np.errstate(over="ignore", under="ignore"):
            estimate = np.exp(estimate)
            interval = np.exp(interval)
        if (
            np.any(~np.isfinite(estimate))
            or np.any(estimate <= 0)
            or np.any(~np.isfinite(interval))
            or np.any(interval <= 0)
        ):
            raise ArithmeticError("raw interaction index is not representable; use log_index=True")

    order = np.argsort(x, kind="stable") if kind == "line" else np.arange(x.size)
    x, estimate, interval = x[order], estimate[order], interval[order]
    yerr: np.ndarray | None = None
    if kind == "points":
        with np.errstate(over="ignore", invalid="ignore"):
            yerr = np.vstack((estimate - interval[:, 0], interval[:, 1] - estimate))
        if np.any(~np.isfinite(yerr)) or np.any(yerr < 0):
            raise ArithmeticError("pointwise interval widths are not representable")
    axes = _axes(ax)
    additivity = 0.0 if log_index else 1.0
    legend_labels = axes.get_legend_handles_labels()[1] if ax is not None else []
    additivity_label = None if "Additivity" in legend_labels else "Additivity"
    if kind == "line":
        (estimate_line,) = axes.plot(x, estimate, label=label)
        axes.fill_between(
            x,
            interval[:, 0],
            interval[:, 1],
            alpha=0.2,
            color=estimate_line.get_color(),
            label=f"{100 * result.confidence:g}% pointwise interval",
        )
    else:
        point_label = (
            f"{label} ({100 * result.confidence:g}% pointwise interval)"
            if label is not None
            else f"Interaction index ({100 * result.confidence:g}% pointwise interval)"
        )
        axes.errorbar(
            x,
            estimate,
            yerr=yerr,
            fmt="o",
            label=point_label,
        )
    axes.axhline(additivity, color="0.35", linestyle=":", label=additivity_label)
    axes.set(
        xlabel="Effect",
        ylabel="Log interaction index" if log_index else "Interaction index",
    )
    return axes
