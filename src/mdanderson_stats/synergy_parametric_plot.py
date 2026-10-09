"""Dose-response and contour figures for source-defined SYNERGY fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .synergy_parametric import SynergyParametricFit, predict_synergy_parametric
from .wfmm_model import _finite_real


def plot_synergy_parametric(
    fit: SynergyParametricFit,
    *,
    dose_ratio: float = 1.0,
    contour_levels: ArrayLike = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    resolution: int = 60,
):
    """Plot observed/fitted single-drug and fixed-ratio curves plus a contour.

    The ratio is dose2/dose1 and mixture curves use total dose on the x axis.
    Observations within 1e-4 of that ratio are shown, as in the author scripts.
    The caller owns the returned matplotlib Figure and can save or close it.
    """
    if not isinstance(fit, SynergyParametricFit):
        raise ValueError("fit must be a SynergyParametricFit")
    if isinstance(dose_ratio, (bool, np.bool_)) or not np.isscalar(dose_ratio):
        raise ValueError("dose_ratio must be a finite positive scalar")
    ratio = float(_finite_real(dose_ratio, "dose_ratio"))
    if not 0 < ratio <= 1e6:
        raise ValueError("dose_ratio must lie in (0,1000000]")
    if (
        isinstance(resolution, (bool, np.bool_))
        or not isinstance(resolution, (int, np.integer))
        or not 10 <= resolution <= 150
    ):
        raise ValueError("resolution must be an integer from 10 to 150")
    levels = _finite_real(contour_levels, "contour_levels")
    if (
        levels.ndim != 1
        or not 1 <= levels.size <= 20
        or np.any((levels <= 0) | (levels >= 1))
        or np.any(np.diff(levels) <= 0)
    ):
        raise ValueError("contour_levels must contain 1 to 20 increasing values in (0,1)")
    fixed = np.abs(fit.dose2 - ratio * fit.dose1) < 1e-4
    maximum = max(
        float(fit.dose1.max()),
        float(fit.dose2.max()),
        float((fit.dose1 + fit.dose2)[fixed].max()) if np.any(fixed) else 0.0,
    )
    total = np.linspace(0, maximum, resolution)
    first = predict_synergy_parametric(fit, total, 0.0)
    second = predict_synergy_parametric(fit, 0.0, total)
    mixture = predict_synergy_parametric(fit, total / (1 + ratio), total * ratio / (1 + ratio))
    x = np.linspace(0, float(fit.dose1.max()), resolution)
    y = np.linspace(0, float(fit.dose2.max()), resolution)
    surface = predict_synergy_parametric(fit, x[None, :], y[:, None])
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("install mdanderson-stats[plot] for SYNERGY figures") from exc
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        for values, label, style in [
            (first, "Drug 1", "--"),
            (second, "Drug 2", ":"),
            (mixture, f"Mixture (dose2/dose1={ratio:g})", "-"),
        ]:
            axes[0].plot(total, values, linestyle=style, label=label)
        axes[0].scatter(fit.dose1[fit.dose2 == 0], fit.response[fit.dose2 == 0], marker="o")
        axes[0].scatter(fit.dose2[fit.dose1 == 0], fit.response[fit.dose1 == 0], marker="s")
        axes[0].scatter((fit.dose1 + fit.dose2)[fixed], fit.response[fixed], marker="x")
        axes[0].set(
            xlabel="Dose (total dose for mixture)", ylabel="Fraction surviving", ylim=(0, 1)
        )
        axes[0].legend()
        contour = axes[1].contour(x, y, surface, levels=levels)
        axes[1].clabel(contour, inline=True, fontsize=8)
        axes[1].set(
            xlabel="Drug 1 dose", ylabel="Drug 2 dose", title=f"{fit.model.title()} response"
        )
        figure.tight_layout()
    except BaseException:
        plt.close(figure)
        raise
    return figure
