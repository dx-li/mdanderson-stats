"""Covariate selection linked to EXPSURV censored survival boxes."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .expsurv_box import CensoredBox, censored_box
from .expsurv_scatter import _ScatterSelection


class CensoredBoxPlot(_ScatterSelection):
    """Retain this controller while interacting with its covariate matrix."""

    def __init__(
        self,
        time: ArrayLike,
        status: ArrayLike,
        covariates: ArrayLike,
        *,
        labels: list[str] | None = None,
        legacy: bool = False,
        quantile_method: Literal["source", "step"] = "source",
    ) -> None:
        baseline = censored_box(time, status, legacy=legacy, quantile_method=quantile_method)
        self.time, self.status = finite(time, "time").copy(), count(status, "status").copy()
        self.time.flags.writeable = self.status.flags.writeable = False
        self.legacy, self.quantile_method = bool(legacy), quantile_method
        self.summary: CensoredBox | None = baseline
        super().__init__(covariates, self.time.size, labels=labels)
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        self.box_figure, self.box_axes = plt.subplots(figsize=(5, 5), layout="constrained")
        upper = float(self.time.max()) or 1.0
        padded = upper * 1.08
        if np.isfinite(padded):
            upper = padded
        self.box_axes.set(xlim=(0, 2), ylim=(0, upper), ylabel="Follow-up time", xticks=[])
        self.segments = LineCollection([], colors="#1675b8", linewidths=1.5)
        self.box_axes.add_collection(self.segments)
        self.annotation = self.box_axes.text(1, 0, "", ha="center", va="bottom")
        self.select(np.arange(self.time.size))

    def _update_selection(self, chosen: np.ndarray) -> None:
        summary = (
            censored_box(
                self.time[chosen],
                self.status[chosen],
                legacy=self.legacy,
                quantile_method=self.quantile_method,
            )
            if chosen.size
            else None
        )
        self.summary = summary
        self.segments.set_segments([] if summary is None else list(summary.segments))
        if summary is None:
            self.annotation.set_text("No selection")
            self.annotation.set_position((1, self.box_axes.get_ylim()[1] / 2))
        elif summary.last_failure_time is None:
            self.annotation.set_text("No observed failures")
            self.annotation.set_position((1, self.box_axes.get_ylim()[1] / 2))
        else:
            self.annotation.set_text(f"S(last failure) = {summary.last_failure_survival:.3g}")
            self.annotation.set_position((1, summary.last_failure_time))
        self.box_axes.set_title(f"Censored survival box: n={chosen.size}")
        self.box_figure.canvas.draw_idle()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        super().close()
        plt.close(self.box_figure)


def plot_censored_box(
    time: ArrayLike,
    status: ArrayLike,
    covariates: ArrayLike,
    *,
    labels: list[str] | None = None,
    legacy: bool = False,
    quantile_method: Literal["source", "step"] = "source",
) -> CensoredBoxPlot:
    """Link matrix row selection to a life-table box; no show/save or backend change."""
    return CensoredBoxPlot(
        time, status, covariates, labels=labels, legacy=legacy, quantile_method=quantile_method
    )
