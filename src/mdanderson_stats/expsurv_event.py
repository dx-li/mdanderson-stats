"""EXPSURV SCAT-EVENT: selected follow-up durations against arrival times."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .expsurv_scatter import _ScatterSelection


class EventScatterPlot(_ScatterSelection):
    """Linked matrix and event chart; retain this controller during interaction."""

    def __init__(
        self,
        arrival: ArrayLike,
        duration: ArrayLike,
        status: ArrayLike,
        covariates: ArrayLike,
        *,
        labels: list[str] | None = None,
    ) -> None:
        a, t, d = (
            finite(arrival, "arrival").copy(),
            finite(duration, "duration").copy(),
            count(status, "status").copy(),
        )
        if a.ndim != 1 or not a.size or np.any(a < 0):
            raise ValueError("arrival must be a nonempty nonnegative vector")
        if t.shape != a.shape or np.any(t < 0):
            raise ValueError("duration must be nonnegative and match arrival")
        if d.shape != a.shape or np.any(d > 1):
            raise ValueError("status must match arrival and contain zero or one")
        self.arrival, self.duration, self.status = a, t, d
        for array in (a, t, d):
            array.flags.writeable = False
        super().__init__(covariates, a.size, labels=labels)
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        self.event_figure, self.event_axes = plt.subplots(layout="constrained")
        self.event_axes.set(
            xlabel="Follow-up duration",
            ylabel="Arrival time",
            xlim=(0, float(t.max()) if t.max() > 0 else 1),
            ylim=(0, float(a.max()) if a.max() > 0 else 1),
        )
        self.segments = LineCollection([], colors="#1675b8", linewidths=1)
        self.event_axes.add_collection(self.segments)
        self.failures = self.event_axes.scatter(
            [], [], marker="+", color="#1675b8", label="Failure", zorder=3, clip_on=False
        )
        self.censored = self.event_axes.scatter(
            [],
            [],
            marker="D",
            facecolors="none",
            edgecolors="#1675b8",
            label="Censored",
            zorder=3,
            clip_on=False,
        )
        self.event_axes.legend()
        self.select(np.arange(a.size))

    def _update_selection(self, chosen: np.ndarray) -> None:
        a, t, d = self.arrival[chosen], self.duration[chosen], self.status[chosen]
        segments = np.empty((chosen.size, 2, 2))
        segments[:, 0, 0] = 0
        segments[:, 1, 0] = t
        segments[:, :, 1] = a[:, None]
        self.segments.set_segments(list(segments))
        endpoints = np.column_stack((t, a))
        self.failures.set_offsets(endpoints[d == 1])
        self.censored.set_offsets(endpoints[d == 0])
        self.event_axes.set_title(f"Selected event chart: n={chosen.size}")
        self.event_figure.canvas.draw_idle()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        super().close()
        plt.close(self.event_figure)


def plot_event_scatter(
    arrival: ArrayLike,
    duration: ArrayLike,
    status: ArrayLike,
    covariates: ArrayLike,
    *,
    labels: list[str] | None = None,
) -> EventScatterPlot:
    """Link matrix selection to horizontal follow-up segments at arrival times.

    Status is one for failure, zero for censoring. All rows start selected.
    Drag replaces, Shift-drag adds, Escape clears; select() uses original row IDs.
    """
    return EventScatterPlot(arrival, duration, status, covariates, labels=labels)
