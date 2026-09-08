"""Linked covariate scatterplot matrix and selected-sample survival curve."""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .expsurv import ExploratorySurvival, exploratory_survival


class _ScatterSelection:
    """Shared matrix selection and callback lifecycle for linked EXPSURV views."""

    def __init__(
        self,
        covariates: ArrayLike,
        n_observations: int,
        *,
        labels: list[str] | None = None,
    ) -> None:
        x = finite(covariates, "covariates").copy()
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != n_observations or x.shape[1] == 0:
            raise ValueError("covariates must have one row per observation and at least one column")
        p = x.shape[1]
        names = [f"Covariate {i + 1}" for i in range(p)] if labels is None else list(labels)
        if len(names) != p or any(not isinstance(s, str) or not s for s in names):
            raise ValueError("labels must contain one nonempty string per covariate")
        from matplotlib import pyplot as plt
        from matplotlib.backend_bases import MouseButton
        from matplotlib.widgets import RectangleSelector

        self.covariates = x
        x.flags.writeable = False
        self.selected_indices = np.empty(0, dtype=np.int64)
        self.selected_indices.flags.writeable = False
        self.figure, self.axes = plt.subplots(
            p, p, squeeze=False, figsize=(max(4, 2.5 * p), max(4, 2.5 * p)), layout="constrained"
        )
        self.figure.suptitle("Drag to select; Shift-drag to add; Escape to clear")
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

    def select(self, indices: ArrayLike, *, add: bool = False) -> None:
        """Select original input row indices, replacing or adding to the current set."""
        values = count(indices, "indices")
        if values.ndim != 1 or np.any(values >= self.covariates.shape[0]):
            raise ValueError("indices must be a vector of valid original row indices")
        if not isinstance(add, (bool, np.bool_)):
            raise ValueError("add must be boolean")
        chosen = np.unique(values.astype(np.int64))
        if add:
            chosen = np.union1d(chosen, self.selected_indices)
        self._update_selection(chosen)
        chosen.flags.writeable = False
        self.selected_indices = chosen
        colors = np.full(self.covariates.shape[0], "#c5c5c5", dtype="<U7")
        colors[chosen] = "#1675b8"
        for points in self.points:
            points.set_facecolors(colors)
            points.set_edgecolors(colors)
        self.figure.canvas.draw_idle()

    def _update_selection(self, chosen: np.ndarray) -> None:
        raise NotImplementedError

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


class SurvivalScatterPlot(_ScatterSelection):
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
        self.time = finite(time, "time").copy()
        self.status = count(status, "status").copy()
        self.time.flags.writeable = self.status.flags.writeable = False
        self.legacy = bool(legacy)
        self.curve: ExploratorySurvival | None = baseline
        super().__init__(covariates, self.time.size, labels=labels)
        from matplotlib import pyplot as plt

        self.survival_figure, self.survival_axes = plt.subplots(layout="constrained")
        self.survival_axes.set(
            xlabel="Follow-up time",
            ylabel="Survival",
            ylim=(0, 1),
            xlim=(0, float(self.time.max()) if self.time.max() > 0 else 1),
        )
        (self.line,) = self.survival_axes.plot([], [])
        self.select(np.arange(self.time.size))

    def _update_selection(self, chosen: np.ndarray) -> None:
        curve = (
            None
            if chosen.size == 0
            else exploratory_survival(self.time[chosen], self.status[chosen], legacy=self.legacy)
        )
        self.curve = curve
        if curve is None:
            self.line.set_data([], [])
        else:
            self.line.set_data(curve.step_time, curve.step_survival)
        self.survival_axes.set_title(f"Selected survival: n={chosen.size}")
        self.survival_figure.canvas.draw_idle()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        super().close()
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
