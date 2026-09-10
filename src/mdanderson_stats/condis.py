"""Conditional-survival restricted-mean imputation for censored lifetimes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .expsurv import ExploratorySurvival, exploratory_survival


@dataclass(frozen=True)
class CondiSImputation:
    observed_time: FloatArray
    status: NDArray[np.int64]
    imputed_time: FloatArray
    remaining_time: FloatArray
    horizon: float
    interpolation: str
    curve: ExploratorySurvival


def condis_impute(
    time: ArrayLike,
    status: ArrayLike,
    *,
    horizon: float | None = None,
    interpolation: str = "linear",
) -> CondiSImputation:
    """Impute c + integral_c^horizon S(t)dt/S(c) for right-censored observations.

    Linear interpolation matches CondiS 0.1.2; step uses the KM step function.
    Observed events and censorings beyond the horizon retain their input times.
    No extrapolation beyond maximum observed follow-up is performed.
    """
    t, d = finite(time, "time"), count(status, "status")
    if t.ndim != 1 or not 1 <= t.size <= 1000000 or d.shape != t.shape:
        raise ValueError("time and status must be matching vectors of length 1..1000000")
    if np.any(t < 0) or np.any(d > 1):
        raise ValueError("time must be nonnegative and status must be 0=censored or 1=event")
    if interpolation not in ("linear", "step"):
        raise ValueError("interpolation must be linear or step")
    maximum = float(t.max())
    endpoint = maximum if horizon is None else scalar(horizon, "horizon")
    if not 0 <= endpoint <= maximum:
        raise ValueError("horizon must lie between zero and maximum observed follow-up")
    curve = exploratory_survival(t, d)
    knots, survival = curve.time, curve.survival
    # Normalize widths, not time coordinates, to retain close differences.
    right = np.minimum(knots[1:], endpoint)
    width = np.maximum(right - knots[:-1], 0)
    scale = endpoint if endpoint else 1.0
    normalized_width = width / scale
    height = survival[:-1]
    if interpolation == "linear":
        fraction = width / np.diff(knots)
        right_survival = survival[:-1] + (survival[1:] - survival[:-1]) * fraction
        height = 0.5 * survival[:-1] + 0.5 * right_survival
    areas = normalized_width * height
    suffix = np.r_[np.cumsum(areas[::-1])[::-1], 0.0]
    at = np.searchsorted(knots, t)
    censored = (d == 0) & (t < endpoint)
    remaining = np.zeros(t.shape)
    denominator = survival[at[censored]]
    if np.any(denominator <= 0):
        raise ArithmeticError("conditional survival is undefined at a censored observation")
    remaining[censored] = (suffix[at[censored]] / denominator) * scale
    # The integral cannot exceed the remaining horizon, including rounding noise.
    remaining[censored] = np.minimum(remaining[censored], endpoint - t[censored])
    imputed = t + remaining
    return CondiSImputation(
        _owned(t),
        _owned(d.astype(np.int64)),
        _owned(imputed),
        _owned(remaining),
        endpoint,
        interpolation,
        curve,
    )
