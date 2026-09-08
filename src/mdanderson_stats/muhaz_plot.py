"""Optional Matplotlib hazard plots and overlays for the MUHAZ estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .kphaz import KPHazard
from .muhaz import MuhazFixed
from .muhaz_global import MuhazGlobal
from .muhaz_knn import MuhazKNN
from .muhaz_local import MuhazLocal
from .pehaz import PiecewiseHazard

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _axes(ax: Axes | None, xlabel: str, ylabel: str) -> Axes:
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(layout="constrained")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    return ax


def plot_muhaz(
    fit: MuhazFixed | MuhazGlobal | MuhazLocal | MuhazKNN,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    xlabel: str = "Follow-up Time",
    ylabel: str = "Hazard Rate",
) -> Axes:
    """Plot kernel hazards, or overlay on ax (the archived lines.muhaz workflow).

    Requires the plot extra. Returns axes without showing/saving the figure or
    changing global style/backend. Supplied axes retain labels and axis settings.
    """
    if not isinstance(fit, (MuhazFixed, MuhazGlobal, MuhazLocal, MuhazKNN)):
        raise TypeError("fit must be a MUHAZ kernel hazard result")
    if fit.time.size == 0:
        raise ValueError("No hazard grid points to plot")
    new = ax is None
    ax = _axes(ax, xlabel, ylabel)
    ax.plot(fit.time, fit.hazard, label=label, color=color, linestyle=linestyle)
    if new:
        ax.set_ylim(0, float(fit.hazard.max()) if fit.hazard.any() else 1)
    return ax


def plot_pehaz(
    fit: PiecewiseHazard,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "--",
    xlabel: str = "Time",
    ylabel: str = "Hazard Rate",
) -> Axes:
    """Plot constant hazards over their bin edges; nonfinite bins are gaps.

    Supplying ax implements lines.pehaz overlays. No artificial drop to zero is
    drawn at either endpoint. All-nonfinite estimates raise ValueError.
    """
    if not isinstance(fit, PiecewiseHazard):
        raise TypeError("fit must be a piecewise-exponential hazard result")
    finite = np.isfinite(fit.hazard)
    if not finite.any():
        raise ValueError("No finite piecewise hazard estimates to plot")
    new = ax is None
    ax = _axes(ax, xlabel, ylabel)
    ax.stairs(
        np.where(finite, fit.hazard, np.nan),
        fit.cuts,
        baseline=None,
        label=label,
        color=color,
        linestyle=linestyle,
    )
    if new:
        ax.set_ylim(0, float(fit.hazard[finite].max()) if fit.hazard[finite].any() else 1)
    return ax


def plot_kphaz(
    fit: KPHazard,
    *,
    ax: Axes | None = None,
    xlabel: str = "Time",
    ylabel: str = "Hazard",
    legend: bool = True,
) -> Axes:
    """Plot failure-interval hazards by stratum, with nonfinite values as gaps.

    Starts at zero when a stratum's first estimate is finite and at positive
    time, as in the archived plot. Strata with no finite estimates are omitted;
    if none remain, raises ValueError. All strata use the same missing-data rule.
    """
    if not isinstance(fit, KPHazard):
        raise TypeError("fit must be a failure-interval hazard result")
    if not isinstance(legend, (bool, np.bool_)):
        raise ValueError("legend must be boolean")
    finite = np.isfinite(fit.hazard)
    if not finite.any():
        raise ValueError("No finite failure-interval hazards to plot")
    new = ax is None
    ax = _axes(ax, xlabel, ylabel)
    styles = ("-", "--", "-.", ":")
    for i, name in enumerate(fit.strata):
        selected = fit.stratum_index == i
        if not finite[selected].any():
            continue
        x = fit.time[selected]
        y = np.where(finite[selected], fit.hazard[selected], np.nan)
        if x[0] > 0 and np.isfinite(y[0]):
            x, y = np.r_[0, x], np.r_[0, y]
        ax.step(x, y, where="post", label=str(name), linestyle=styles[i % len(styles)])
    if new:
        ax.set_ylim(0, float(fit.hazard[finite].max()) if fit.hazard[finite].any() else 1)
    if legend:
        ax.legend()
    return ax
