"""Observed and imputed Kaplan--Meier comparison for CondiS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import finite
from .condis import CondiSImputation
from .expsurv import ExploratorySurvival, exploratory_survival

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class CondiSSurvivalComparison:
    """Two CondiS vignette curves and an explicit, caller-timed risk table.

    ``observed_curve`` uses original censoring indicators. ``imputed_curve``
    treats each imputed time as an event, matching the CondiS vignette; it is a
    descriptive visualization and does not provide censoring-based inference.
    Risk counts use the pre-event convention ``time >= risk_time``.
    """

    observed_curve: ExploratorySurvival
    imputed_curve: ExploratorySurvival
    censor_time: np.ndarray
    censor_survival: np.ndarray
    risk_times: np.ndarray
    observed_at_risk: np.ndarray
    imputed_at_risk: np.ndarray


def _freeze_int(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def condis_survival_comparison(
    fit: CondiSImputation, *, risk_times: ArrayLike
) -> CondiSSurvivalComparison:
    """Reproduce the CondiS vignette's censored-versus-imputed KM comparison.

    ``risk_times`` is explicit rather than guessed from plotting-library
    defaults. At each supplied time, a subject is counted at risk when their
    observed or imputed time is greater than or equal to that time.
    """

    if not isinstance(fit, CondiSImputation):
        raise TypeError("fit must be a CondiSImputation")
    time = finite(fit.observed_time, "observed_time")
    status = finite(fit.status, "status")
    imputed = finite(fit.imputed_time, "imputed_time")
    if (
        time.ndim != 1
        or time.size < 1
        or time.size > 1_000_000
        or status.shape != time.shape
        or imputed.shape != time.shape
    ):
        raise ValueError("fit vectors must be matching vectors of length 1..1000000")
    if (
        np.any(time < 0)
        or np.any(imputed < 0)
        or np.any((status != 0) & (status != 1))
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(imputed))
    ):
        raise ValueError("fit times must be finite and nonnegative and status must be 0 or 1")

    if isinstance(risk_times, np.ndarray):
        if risk_times.ndim != 1 or risk_times.size > 20:
            raise ValueError("risk_times must contain at most 20 values")
    elif isinstance(risk_times, (list, tuple)):
        if len(risk_times) > 20 or any(
            isinstance(value, (list, tuple, np.ndarray)) for value in risk_times
        ):
            raise ValueError("risk_times must be a flat sequence of at most 20 values")
    times = finite(risk_times, "risk_times")
    if (
        times.ndim != 1
        or times.size < 1
        or times.size > 20
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("risk_times must be 1..20 finite, nonnegative, strictly increasing values")

    observed_curve = exploratory_survival(time, status)
    imputed_curve = exploratory_survival(imputed, np.ones(time.size, dtype=np.int64))
    censored = status == 0
    censor_time = time[censored]
    censor_survival = observed_curve.at(censor_time)
    observed_risk = time.size - np.searchsorted(np.sort(time), times, side="left")
    imputed_risk = imputed.size - np.searchsorted(np.sort(imputed), times, side="left")
    return CondiSSurvivalComparison(
        observed_curve,
        imputed_curve,
        _freeze(censor_time),
        _freeze(censor_survival),
        _freeze(times),
        _freeze_int(observed_risk),
        _freeze_int(imputed_risk),
    )


def plot_condis_survival_comparison(
    comparison: CondiSSurvivalComparison,
    *,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Axes, Axes]:
    """Plot the two curves, censor marks and risk counts using lazy Matplotlib."""

    if not isinstance(comparison, CondiSSurvivalComparison):
        raise TypeError("comparison must be a CondiSSurvivalComparison")
    from matplotlib import pyplot as plt

    if axes is None:
        figure, (curve_ax, risk_ax) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        figure.subplots_adjust(hspace=0.08)
    else:
        if not isinstance(axes, tuple) or len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise ValueError("axes must be a pair of axes belonging to the same figure")
        curve_ax, risk_ax = axes

    observed, imputed = comparison.observed_curve, comparison.imputed_curve
    curve_ax.step(observed.step_time, observed.step_survival, where="post", label="Censored")
    curve_ax.step(imputed.step_time, imputed.step_survival, where="post", label="CondiS imputed")
    if comparison.censor_time.size:
        curve_ax.scatter(
            comparison.censor_time,
            comparison.censor_survival,
            marker="+",
            color="C0",
            label="Censoring",
            zorder=3,
        )
    curve_ax.set(xlabel="Follow-up time", ylabel="Survival probability", ylim=(0, 1.02))
    curve_ax.tick_params(axis="x", labelbottom=True)
    curve_ax.legend()
    risk_ax.axis("off")
    header = [f"{value:g}" for value in comparison.risk_times]
    rows = [
        ["At risk: censored", *map(str, comparison.observed_at_risk.astype(int))],
        ["At risk: imputed", *map(str, comparison.imputed_at_risk.astype(int))],
    ]
    risk_ax.table(cellText=rows, colLabels=["", *header], loc="center")
    return curve_ax, risk_ax
