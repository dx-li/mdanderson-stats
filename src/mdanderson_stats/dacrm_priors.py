"""Convenience elicitation helpers for DA-CRM piecewise hazard priors."""

from math import sqrt
from typing import Literal

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
    endpoint_policy: Literal["fixed", "native_desktop"] = "fixed",
    alpha_sd: float = sqrt(2),
) -> DACRMPrior:
    """Build six-piece hazards calibrated to three trimester DLT probabilities.

    The three positive probabilities must sum to one. The final intermediate
    cumulative probability must be below 0.99 so the six calibrated hazards
    remain finite and positive. The guide does not specify a gamma dispersion,
    so callers must supply it explicitly. Recovered desktop code uses dispersion
    two and ``endpoint_policy="native_desktop"``: for any interval whose
    unadjusted next cumulative probability is within 1e-6 of one, use .99 when
    the previous value is below .99, otherwise .9 + .1 times that previous value.
    The default fixed endpoint preserves the original Python API.
    """
    if endpoint_policy not in {"fixed", "native_desktop"}:
        raise ValueError("endpoint_policy must be 'fixed' or 'native_desktop'")
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
    if endpoint_policy == "native_desktop":
        cumulative[-1] = p1 + p2 + p3
    previous, following = cumulative[:-1], cumulative[1:].copy()
    if endpoint_policy == "native_desktop":
        # Native code calibrates each interval separately: adjusting q_next
        # does not alter the following interval's original q_previous.
        near_one = np.abs(following - 1) < 1e-6
        following[near_one] = np.where(
            previous[near_one] < 0.99, 0.99, 0.9 + 0.1 * previous[near_one]
        )
    if np.any(following <= previous) or np.any(following >= 1):
        raise ValueError(
            "trimester cumulative probabilities must increase strictly "
            "within each interval below one"
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        means = (6.0 / window_value) * (np.log1p(-previous) - np.log1p(-following))
    if np.any(~np.isfinite(means) | (means <= 0)):
        raise ValueError("trimester probabilities imply nonpositive or unrepresentable hazards")
    return _prior(window_value, means, dispersion_value, alpha_sd)
