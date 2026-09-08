"""CUMINC overlays and cause-by-group panels with pointwise intervals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite
from .cuminc_study import CumIncStudy, _select

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_cuminc(
    study: CumIncStudy,
    *,
    causes: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    overlay: bool = True,
    xlabel: str = "Time",
    ylabel: str = "Cumulative incidence",
    legend_at: ArrayLike | None = None,
) -> tuple[Axes, ...]:
    """Plot selected curves together, or individually with confidence limits.

    Return axes in cause-then-group order (one axis for an overlay). For one
    selected group, panels are arranged horizontally by cause; otherwise rows
    are causes and columns groups. legend_at gives the overlay legend's upper
    left corner in data coordinates. Requires the optional plot extra. Does not
    show/save the figure or change global style/backend. Returned axes support
    further Matplotlib customization and figure saving.
    """
    if not isinstance(overlay, (bool, np.bool_)):
        raise ValueError("overlay must be boolean")
    selected_causes = _select(causes, study.causes, "causes")
    selected_groups = _select(groups, study.groups, "groups")
    if not selected_causes or not selected_groups:
        raise ValueError("At least one cause and group must be selected for plotting")
    location = None
    if legend_at is not None:
        location = finite(legend_at, "legend_at")
        if location.shape != (2,):
            raise ValueError("legend_at must be two finite data coordinates")
        if not overlay:
            raise ValueError("legend_at applies only to overlay plots")
    keys = [(c, g) for c in selected_causes for g in selected_groups]
    if overlay:
        nrows = ncols = 1
    elif len(selected_groups) == 1:
        nrows, ncols = 1, len(selected_causes)
    else:
        nrows, ncols = len(selected_causes), len(selected_groups)
    from matplotlib import pyplot as plt

    _, grid = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(5 * ncols, 3.8 * nrows), layout="constrained"
    )
    axes = tuple(grid.ravel())
    styles = ("-", "--", "-.", ":")
    for index, key in enumerate(keys):
        curve = study.curves[key]
        ax = axes[0] if overlay else axes[index]
        label = f"Cause = {key[0]}, Group = {key[1]}"
        # CINC already contains paired left/right corners: ordinary lines retain
        # exact vertical jumps, including a jump at zero, without interpolation.
        ax.plot(
            curve.time,
            curve.estimate,
            linestyle=styles[index % 4] if overlay else "-",
            label=label if overlay else "Incidence",
        )
        if not overlay:
            summary = curve.summary(confidence=study.confidence)
            percent = str(100 * study.confidence).removesuffix(".0")
            ax.plot(curve.time, summary.rows[:, 3], "--", color="0.4", label=f"{percent}% limits")
            ax.plot(curve.time, summary.rows[:, 4], "--", color="0.4")
            ax.set_title(label)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, curve.time[-1] if curve.time[-1] > 0 else 1)
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if overlay:
        ax = axes[0]
        max_time = max(study.curves[key].time[-1] for key in keys)
        max_incidence = max(float(np.max(study.curves[key].estimate)) for key in keys)
        ax.set_xlim(0, max_time if max_time > 0 else 1)
        ax.set_ylim(0, max_incidence if max_incidence > 0 else 1)
        ax.set_title("Cumulative incidence curves")
        if location is None:
            ax.legend()
        else:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=tuple(location),
                bbox_transform=ax.transData,
                borderaxespad=0,
            )
    return axes
