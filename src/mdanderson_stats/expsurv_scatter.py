"""Linked covariate scatterplot matrix and selected-sample survival curve."""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .expsurv import ExploratorySurvival, exploratory_survival


class SurvivalScatterPlot:
    """Retain this controller while selecting patients in its matrix figure."""

    def __init__(
        self,
        time: ArrayLike,
        status: ArrayLike,
        covariates: ArrayLike,
        *,
        labels: list[str] | None = None,
        legacy: bool = False,
    ) -> None:
        baseline = exploratory_survival(time, status, legacy=legacy)
        t, d, x = (
            finite(time, "time").copy(),
            count(status, "status").copy(),
            finite(covariates, "covariates").copy(),
        )
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != t.size or x.shape[1] == 0:
            raise ValueError("covariates must have one row per observation and at least one column")
        p = x.shape[1]
        names = [f"Covariate {i + 1}" for i in range(p)] if labels is None else list(labels)
        if len(names) != p or any(not isinstance(s, str) or not s for s in names):
            raise ValueError("labels must contain one nonempty string per covariate")
        from matplotlib import pyplot as plt
        from matplotlib.backend_bases import MouseButton
        from matplotlib.widgets import RectangleSelector

        self.time, self.status, self.covariates = t, d, x
        for array in (t, d, x):
            array.flags.writeable = False
        self.legacy = bool(legacy)
        self.curve: ExploratorySurvival | None = baseline
        self.selected_indices = np.arange(t.size)
        self.figure, self.axes = plt.subplots(
            p, p, squeeze=False, figsize=(max(4, 2.5 * p), max(4, 2.5 * p)), layout="constrained"
        )
        self.figure.suptitle("Drag to select; Shift-drag to add; Escape to clear")
        self.survival_figure, self.survival_axes = plt.subplots(layout="constrained")
        self.survival_axes.set(
            xlabel="Follow-up time",
            ylabel="Survival",
            ylim=(0, 1),
            xlim=(0, float(t.max()) if t.max() > 0 else 1),
        )
        (self.line,) = self.survival_axes.plot([], [])
        self.points = []
        self.selectors = []
        for row in range(p):
            for col in range(p):
                ax = self.axes[row, col]
                self.points.append(ax.scatter(x[:, col], x[:, row], s=22))
                if row == p - 1:
                    ax.set_xlabel(names[col])
                if col == 0:
                    ax.set_ylabel(names[row])
                self.selectors.append(
                    RectangleSelector(
                        ax,
                        partial(self._rectangle, row, col),
                        button=[MouseButton.LEFT],
                        interactive=True,
                        state_modifier_keys={"square": "alt"},
                    )
                )
        self._key_callback = self.figure.canvas.mpl_connect("key_press_event", self._key)
        self.select(np.arange(t.size))

    def select(self, indices: ArrayLike, *, add: bool = False) -> None:
        """Select original input row indices, replacing or adding to the current set."""
        values = count(indices, "indices")
        if values.ndim != 1 or np.any(values >= self.time.size):
            raise ValueError("indices must be a vector of valid original row indices")
        if not isinstance(add, (bool, np.bool_)):
            raise ValueError("add must be boolean")
        chosen = np.unique(values.astype(np.int64))
        if add:
            chosen = np.union1d(chosen, self.selected_indices)
        curve = (
            None
            if chosen.size == 0
            else exploratory_survival(self.time[chosen], self.status[chosen], legacy=self.legacy)
        )
        chosen.flags.writeable = False
        self.selected_indices, self.curve = chosen, curve
        colors = np.full(self.time.size, "#c5c5c5", dtype="<U7")
        colors[chosen] = "#1675b8"
        for points in self.points:
            points.set_facecolors(colors)
            points.set_edgecolors(colors)
        if curve is None:
            self.line.set_data([], [])
        else:
            self.line.set_data(curve.step_time, curve.step_survival)
        self.survival_axes.set_title(f"Selected survival: n={chosen.size}")
        self.figure.canvas.draw_idle()
        self.survival_figure.canvas.draw_idle()

    def _rectangle(self, row: int, col: int, press, release) -> None:
        coordinates = [press.xdata, release.xdata, press.ydata, release.ydata]
        if any(v is None for v in coordinates):
            return
        xmin, xmax = sorted(coordinates[:2])
        ymin, ymax = sorted(coordinates[2:])
        x, y = self.covariates[:, col], self.covariates[:, row]
        selected = np.flatnonzero((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
        self.select(selected, add=press.key is not None and "shift" in press.key)

    def _key(self, event) -> None:
        if event.key == "escape":
            self.clear()

    def clear(self) -> None:
        self.select([])
        for selector in self.selectors:
            selector.clear()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        for selector in self.selectors:
            selector.disconnect_events()
        self.figure.canvas.mpl_disconnect(self._key_callback)
        plt.close(self.figure)
        plt.close(self.survival_figure)


def plot_survival_scatter(
    time: ArrayLike,
    status: ArrayLike,
    covariates: ArrayLike,
    *,
    labels: list[str] | None = None,
    legacy: bool = False,
) -> SurvivalScatterPlot:
    """Create the EXPSURV SCAT-KM linked views without showing/saving figures.

    Covariates are observations-by-variables. Drag a rectangle in any cell to
    replace selection, Shift-drag to add, Escape to clear. Programmatic select()
    accepts original row indices. No global backend or style is changed.
    """
    return SurvivalScatterPlot(time, status, covariates, labels=labels, legacy=legacy)
