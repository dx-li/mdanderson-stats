"""Interactive cut-point density and survival views using optional Matplotlib."""

from __future__ import annotations

import numpy as np

from ._validation import FloatArray, scalar
from .expsurv_cutpoint import SurvivalCutpoint


def _density(values: FloatArray, bandwidth: float | None) -> tuple[FloatArray, FloatArray, float]:
    """Gaussian guide with explicit bandwidth; bounded query-by-subject storage."""
    h = (
        float(np.std(values, ddof=1) * values.size ** (-0.2))
        if bandwidth is None
        else scalar(bandwidth, "bandwidth")
    )
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Density bandwidth must be finite and positive")
    grid = np.linspace(values.min() - 3 * h, values.max() + 3 * h, 256)
    result = np.empty(grid.size)
    chunk = max(1, 262144 // values.size)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, grid.size, chunk):
                u = (grid[start : start + chunk, None] - values) / h
                result[start : start + chunk] = np.exp(-0.5 * u * u).mean(axis=1) / (
                    h * np.sqrt(2 * np.pi)
                )
    except FloatingPointError as exc:
        raise RuntimeError("Density guide exceeds numerical range") from exc
    return grid, result, h


class CutpointPlot:
    """Keep this controller alive while using its linked slider and figure."""

    def __init__(
        self, data: SurvivalCutpoint, *, cut: float | None = None, bandwidth: float | None = None
    ) -> None:
        if not isinstance(data, SurvivalCutpoint):
            raise TypeError("data must be prepared with survival_cutpoint")
        lo, hi = float(data.covariate.min()), float(data.covariate.max())
        if lo == hi:
            raise ValueError(
                "The slider requires a varying covariate; use compare() for constant data"
            )
        positions = np.linspace(lo, hi, 50)
        initial = float(positions[25]) if cut is None else scalar(cut, "cut")
        if not lo <= initial <= hi:
            raise ValueError("Initial cut must be within the observed covariate range")
        grid, density, h = _density(data.covariate, bandwidth)
        from matplotlib import pyplot as plt
        from matplotlib.widgets import Slider

        self.data = data
        self.comparison = data.compare(initial)
        self.density_bandwidth = h
        self.figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        self.figure.subplots_adjust(bottom=0.25, wspace=0.3)
        self.density_axes, self.survival_axes = axes
        self.density_axes.plot(grid, density)
        self.density_axes.set(xlabel="Covariate", ylabel="Density", title="Cut-point exploration")
        self.cut_line = self.density_axes.axvline(initial, color="black", linestyle="--")
        (self.lower_line,) = self.survival_axes.plot([], [], label="Lower")
        (self.upper_line,) = self.survival_axes.plot([], [], label="Upper")
        self.survival_axes.set(
            xlabel="Follow-up time",
            ylabel="Survival",
            ylim=(0, 1),
            xlim=(0, max(float(data.time.max()), 1)),
        )
        slider_axes = self.figure.add_axes((0.2, 0.08, 0.65, 0.04))
        self.slider = Slider(slider_axes, "Cut", lo, hi, valinit=initial, valstep=positions)
        self._callback_id = self.slider.on_changed(self._update)
        self.slider.set_val(initial)

    def _update(self, cut: float) -> None:
        comparison = self.data.compare(cut)
        for line, curve, indices, symbol in [
            (self.lower_line, comparison.lower, comparison.lower_indices, "≤"),
            (self.upper_line, comparison.upper, comparison.upper_indices, ">"),
        ]:
            line.set_data([], []) if curve is None else line.set_data(
                curve.step_time, curve.step_survival
            )
            line.set_label(f"{symbol} {cut:.4g}: n={indices.size}")
        self.cut_line.set_xdata([cut, cut])
        self.survival_axes.legend()
        self.comparison = comparison
        self.figure.canvas.draw_idle()

    def set_cut(self, cut: float) -> None:
        """Set any finite cut within the observed range; slider dragging snaps to 50 positions."""
        cut = scalar(cut, "cut")
        if not self.slider.valmin <= cut <= self.slider.valmax:
            raise ValueError("Cut must be within the observed covariate range")
        self.slider.set_val(cut)

    def close(self) -> None:
        """Disconnect the slider callback and close its figure."""
        from matplotlib import pyplot as plt

        self.slider.disconnect(self._callback_id)
        plt.close(self.figure)


def plot_cutpoint(
    data: SurvivalCutpoint, *, cut: float | None = None, bandwidth: float | None = None
) -> CutpointPlot:
    """Create linked density/survival panels and a cut slider without calling show.

    The density guide uses a Gaussian kernel with std(ddof=1)*n**(-1/5) bandwidth
    unless supplied explicitly. This replaces XLISP-STAT's implicit kernel-dens
    dependency; it is not claimed to reproduce that runtime's density defaults.
    """
    return CutpointPlot(data, cut=cut, bandwidth=bandwidth)
