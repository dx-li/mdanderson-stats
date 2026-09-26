"""Convenience elicitation helpers for DA-CRM piecewise hazard priors."""

from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from ._validation import scalar
from .dacrm import DACRMPrior


def _positive(value: float, name: str) -> float:
    result = scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _prior(window: float, means: np.ndarray, dispersion: float, alpha_sd: float) -> DACRMPrior:
    intervals = means.size
    breaks = np.linspace(0.0, window, intervals + 1)
    return DACRMPrior(
        breaks,
        means / dispersion,
        np.full(intervals, 1.0 / dispersion),
        alpha_sd,
    )


def dacrm_uniform_prior(
    window: float,
    *,
    intervals: int = 9,
    dispersion: float = 2.0,
    alpha_sd: float = sqrt(2),
) -> DACRMPrior:
    """Build the paper's increasing-mean hazard prior over equal intervals.

    Interval ``k`` has mean hazard ``K / (window * (K-k+0.5))`` for
    one-based ``k`` and Gamma rate ``1/dispersion``. Dispersion therefore
    carries inverse-time units when the time unit changes.
    """
    window_value = _positive(window, "window")
    dispersion_value = _positive(dispersion, "dispersion")
    if isinstance(intervals, (bool, np.bool_)) or not isinstance(intervals, (int, np.integer)):
        raise ValueError("intervals must be an integer from 1 to 20")
    count = int(intervals)
    if not 1 <= count <= 20:
        raise ValueError("intervals must be an integer from 1 to 20")
    positions = np.arange(1, count + 1, dtype=float)
    means = (count / window_value) / (count - positions + 0.5)
    if np.any(~np.isfinite(means)):
        raise ValueError("window is too small to represent prior mean hazards")
    return _prior(window_value, means, dispersion_value, alpha_sd)


def dacrm_trimester_prior(
    window: float,
    probabilities: ArrayLike,
    *,
    dispersion: float,
    alpha_sd: float = sqrt(2),
) -> DACRMPrior:
    """Build six-piece hazards calibrated to three trimester DLT probabilities.

    The three positive probabilities must sum to one. The final intermediate
    cumulative probability must be below 0.99 so the six calibrated hazards
    remain finite and positive. The guide does not specify a gamma dispersion,
    so callers must supply it explicitly.
    """
    window_value = _positive(window, "window")
    dispersion_value = _positive(dispersion, "dispersion")
    raw = np.asarray(probabilities)
    if raw.ndim != 1 or raw.size != 3 or raw.dtype.kind not in "iuf":
        raise ValueError("probabilities must be three real numeric values")
    probs = np.asarray(raw, dtype=float)
    if (
        np.any(~np.isfinite(probs))
        or np.any(probs <= 0)
        or not np.isclose(float(np.sum(probs)), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("probabilities must be positive and sum to one")
    p1, p2, p3 = probs
    cumulative = np.array([0.0, p1 / 2, p1, p1 + p2 / 2, p1 + p2, p1 + p2 + p3 / 2, 0.99])
    if np.any(np.diff(cumulative) <= 0):
        raise ValueError("trimester cumulative probabilities must increase strictly to 0.99")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        means = (6.0 / window_value) * (np.log1p(-cumulative[:-1]) - np.log1p(-cumulative[1:]))
    if np.any(~np.isfinite(means) | (means <= 0)):
        raise ValueError("trimester probabilities imply nonpositive or unrepresentable hazards")
    return _prior(window_value, means, dispersion_value, alpha_sd)
