"""Vectorized one-sample confidence intervals, including BP1CI compatibility.

These are independent implementations of distribution-tail inversion formulas;
no legacy numerical library code is embedded.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainccinv, betaincinv, gammainccinv, gammaincinv

from ._validation import FloatArray
from ._validation import count as _count
from ._validation import finite as _finite


def _tail(confidence: ArrayLike) -> FloatArray:
    array = _finite(confidence, "confidence")
    if np.any((array <= 0) | (array >= 1)):
        raise ValueError("confidence must lie strictly between zero and one")
    return (1 - array) / 2


def binomial_interval(
    successes: ArrayLike, trials: ArrayLike, confidence: ArrayLike = 0.95
) -> tuple[FloatArray, FloatArray]:
    """Equal-tailed Clopper–Pearson bounds; output shape is the broadcast shape.

    For zero trials, return [0, 1] (no information). Confidence is a fraction,
    whereas the original BP1CI interactive program accepts a percentage.
    """
    k, n, tail = np.broadcast_arrays(
        _count(successes, "successes"), _count(trials, "trials"), _tail(confidence)
    )
    if np.any(k > n):
        raise ValueError("successes must not exceed trials")
    lower = np.zeros(k.shape)
    upper = np.ones(k.shape)
    positive = k > 0
    below_n = k < n
    lower[positive] = betaincinv(k[positive], n[positive] - k[positive] + 1, tail[positive])
    upper[below_n] = betainccinv(k[below_n] + 1, n[below_n] - k[below_n], tail[below_n])
    return lower, upper


def _poisson_inputs(
    events: ArrayLike, confidence: ArrayLike, exposure: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    k, tail, time = np.broadcast_arrays(
        _count(events, "events"), _tail(confidence), _finite(exposure, "exposure")
    )
    if np.any(time <= 0):
        raise ValueError("exposure must be positive")
    return k, tail, time


def poisson_interval(
    events: ArrayLike, confidence: ArrayLike = 0.95, exposure: ArrayLike = 1
) -> tuple[FloatArray, FloatArray]:
    """Exact equal-tailed Garwood bounds on the Poisson rate per unit exposure."""
    k, tail, time = _poisson_inputs(events, confidence, exposure)
    lower = np.zeros(k.shape)
    positive = k > 0
    lower[positive] = gammaincinv(k[positive], tail[positive])
    upper = gammainccinv(k + 1, tail)
    return lower / time, upper / time


def bp1ci_poisson_interval(
    events: ArrayLike, confidence: ArrayLike = 0.95, exposure: ArrayLike = 1
) -> tuple[FloatArray, FloatArray]:
    """Reproduce BP1CI 2.0's Poisson formula, including its lower-bound defect.

    The original inverts P(X > k) for the lower bound, rather than P(X >= k).
    Thus its lower gamma shape is k+1 instead of k.
    Use poisson_interval for an exact confidence interval. The exposure argument
    performs the manual post-calculation scaling described in the BP1CI guide.
    """
    k, tail, time = np.broadcast_arrays(
        _finite(events, "events"), _tail(confidence), _finite(exposure, "exposure")
    )
    if np.any(k < 1e-6):
        raise ValueError("BP1CI requires events >= 1e-6 (fractional events are permitted)")
    if np.any(time <= 0):
        raise ValueError("exposure must be positive")
    return gammaincinv(k + 1, tail) / time, gammainccinv(k + 1, tail) / time
