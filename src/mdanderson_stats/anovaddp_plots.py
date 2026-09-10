"""The five supplied ANOVA DDP report figures, with optional physical time axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .anovaddp_io import anovaddp_r_outputs
from .anovaddp_prediction import AnovaDDPPrediction

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_anovaddp(prediction: AnovaDDPPrediction, *, use_time: bool = False) -> tuple[Figure, ...]:
    """Reproduce plotResults using optional Matplotlib; return figures without showing.

    Default x coordinates are 1..T, exactly as in R's implicit plot indices.
    use_time=True labels actual prediction times. Each component page uses a
    2-by-5 grid; more than ten scenarios create additional pages, none are dropped.
    Figure 1 retains the source's 2-by-1 layout with an empty lower panel.
    """
    from matplotlib import pyplot as plt

    if not isinstance(use_time, bool):
        raise ValueError("use_time must be boolean")
    values = anovaddp_r_outputs(prediction)
    x = prediction.time if use_time else np.arange(1, prediction.time.size + 1)
    figures: list[Figure] = []
    fig, axes = plt.subplots(2, 1, figsize=(7.75, 6.5), facecolor="white")
    axes[0].plot(x, values["m"].mean(axis=0), color="blue")
    axes[0].set_title("m")
    if use_time:
        axes[0].set_xlabel("Time")
    axes[1].set_axis_off()
    fig.suptitle("Figure 1")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures.append(fig)
    for number, name in enumerate(("a0", "a02", "f0", "f02"), start=2):
        rows = values[name]
        for start in range(0, rows.shape[0], 10):
            fig, axes = plt.subplots(2, 5, figsize=(7.75, 6.5), facecolor="white")
            for offset, ax in enumerate(axes.flat):
                row = start + offset
                if row >= rows.shape[0]:
                    ax.set_axis_off()
                    continue
                ax.plot(x, rows[row], color="blue")
                ax.set_title(f"{name} {row + 1}")
                if use_time:
                    ax.set_xlabel("Time")
            suffix = f" (page {start // 10 + 1})" if rows.shape[0] > 10 else ""
            fig.suptitle(f"Figure {number}{suffix}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            figures.append(fig)
    return tuple(figures)
