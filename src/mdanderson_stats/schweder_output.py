"""MULTI's S Schweder plot and desktop coordinate export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from .schweder import SchwederFit

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_schweder(fit: SchwederFit, ax: Axes | None = None) -> Axes:
    """Plot all (1-p,Np) points and the origin-to-null-estimate line.

    Requires the optional plot extra. Returns the axes for customization or
    saving; does not call show or change global plotting style/backend.
    As in S schwed.plot, x spans [0,1] and y ticks appear on both sides.
    """
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(layout="constrained")
    ax.plot(fit.one_minus_p, fit.upper_counts, "o", markerfacecolor="none", label="Observed")
    # S plot establishes y limits from the observations before overlaying lines.
    y_limits = ax.get_ylim()
    ax.plot([0, 1], [0, fit.null_estimate], label="Fitted line")
    ax.set_xlim(0, 1)
    ax.set_ylim(y_limits)
    ax.set_xlabel("1-p")
    ax.set_ylabel("Np")
    ax.tick_params(axis="y", right=True, labelright=True)
    return ax


def write_schweder_data(fit: SchwederFit, path: str | Path) -> Path:
    """Write all Schweder coordinates to CSV, replacing path if it exists.

    Columns 1-p and Np correspond to desktop SWFIT's exported coordinates.
    CSV and round-trip float precision replace the original fixed-width format.
    All points are included, not just those used to estimate the line.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["1-p", "Np"])
        writer.writerows(
            (format(float(x), ".17g"), int(count))
            for x, count in zip(fit.one_minus_p, fit.upper_counts, strict=True)
        )
    return path
