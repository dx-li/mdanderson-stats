"""Interactive accelerated-failure and proportional-hazards curve alignment."""

from typing import Literal

import numpy as np

from ._validation import finite
from .expsurv import ExploratorySurvival

AlignmentMethod = Literal["accelerated-failure", "proportional-hazards"]


class SurvivalAlignmentPlot:
    """Retain this controller while interacting with its two sliders."""

    def __init__(
        self,
        first: ExploratorySurvival,
        second: ExploratorySurvival,
        *,
        method: AlignmentMethod = "proportional-hazards",
    ) -> None:
        if not isinstance(first, ExploratorySurvival) or not isinstance(
            second, ExploratorySurvival
        ):
            raise TypeError("Both inputs must be exploratory_survival results")
        if method not in ("accelerated-failure", "proportional-hazards"):
            raise ValueError("Unknown alignment method")
        maximum = np.array([first.time.max(), second.time.max()])
        upper = np.ones(2)
        if method == "accelerated-failure":
            if np.any(maximum <= 0):
                raise ValueError("Time alignment requires positive follow-up in both samples")
            upper = maximum.max() / maximum
            if not np.isfinite(upper).all():
                raise ValueError("Time-scale ratio exceeds numerical range")
        from matplotlib import pyplot as plt
        from matplotlib.widgets import Slider

        self.curves = (first, second)
        self.method = method
        self.figure, self.axes = plt.subplots(figsize=(7, 5))
        self.figure.subplots_adjust(bottom=0.3)
        self.lines = tuple(
            self.axes.plot(c.step_time, c.step_survival, label=f"Sample {i + 1}")[0]
            for i, c in enumerate(self.curves)
        )
        self.axes.set(
            xlim=(0, float(maximum.max()) if maximum.max() > 0 else 1),
            ylim=(0, 1),
            xlabel="Follow-up time",
            ylabel="Survival",
            title=method.replace("-", " ").capitalize(),
        )
        self.axes.legend()
        sliders = []
        for i in range(2):
            ax = self.figure.add_axes((0.23, 0.16 - i * 0.08, 0.65, 0.035))
            sliders.append(
                Slider(
                    ax,
                    f"Sample {i + 1}",
                    0,
                    float(upper[i]),
                    valinit=float(upper[i]),
                    valstep=np.linspace(0, upper[i], 50),
                )
            )
        self.sliders = tuple(sliders)
        self._callbacks = tuple(s.on_changed(self._update) for s in self.sliders)
        self._update(0)

    def _update(self, value: float) -> None:
        for curve, line, slider in zip(self.curves, self.lines, self.sliders, strict=True):
            if self.method == "accelerated-failure":
                line.set_data(curve.step_time * slider.val, curve.step_survival)
            else:
                line.set_data(curve.step_time, np.power(curve.step_survival, slider.val))
        self.figure.canvas.draw_idle()

    def set_parameters(self, first: float, second: float) -> None:
        """Validate both slider parameters before updating either one."""
        values = finite([first, second], "parameters")
        if (
            values.shape != (2,)
            or np.any(values < 0)
            or any(v > s.valmax for v, s in zip(values, self.sliders, strict=True))
        ):
            raise ValueError("Parameters must lie within their slider ranges")
        for slider, value in zip(self.sliders, values, strict=True):
            slider.set_val(float(value))

    def close(self) -> None:
        from matplotlib import pyplot as plt

        for slider, callback in zip(self.sliders, self._callbacks, strict=True):
            slider.disconnect(callback)
        plt.close(self.figure)


def plot_survival_alignment(
    first: ExploratorySurvival,
    second: ExploratorySurvival,
    *,
    method: AlignmentMethod = "proportional-hazards",
) -> SurvivalAlignmentPlot:
    """Explore alignment by x -> multiplier*x or S -> S**power, without refitting.

    Proportional-hazards powers range from 0 to 1. Time multipliers range from
    zero to max(max_time1,max_time2)/max_time_i, matching the archived sliders.
    Zero time multipliers collapse a curve to time zero; zero powers produce a
    curve of ones, including the explicit plotting convention 0**0=1.
    Does not show/save the figure or change the plotting backend.
    """
    return SurvivalAlignmentPlot(first, second, method=method)
