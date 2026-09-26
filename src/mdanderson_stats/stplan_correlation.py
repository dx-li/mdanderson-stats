"""STPLAN Fisher-transform approximations for correlation power."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ._validation import FloatArray
from .stplan_continuous import _alpha, _inputs, _power


def _moments(rho: FloatArray, n: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.arctanh(rho) + rho / (2 * (n - 1))
    reciprocal = 1 / (n - 1)
    variance = reciprocal + (4 - rho * rho) * reciprocal * reciprocal / 2
    return mean, np.sqrt(variance)


def stplan_correlation_one_sample_power(
    null_correlation: ArrayLike,
    alternative_correlation: ArrayLike,
    sample_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Current STPLAN one-sample bivariate-normal correlation approximation."""
    a = _alpha(alpha, sides)
    rho0, rhoa, n = _inputs(
        (null_correlation, "null_correlation"),
        (alternative_correlation, "alternative_correlation"),
        (sample_size, "sample_size"),
    )
    if np.any(np.abs(rho0) >= 1) or np.any(np.abs(rhoa) >= 1) or np.any(n < 4):
        raise ValueError("correlations must be strictly between -1 and 1; sample_size >= 4")
    m0, sd0 = _moments(rho0, n)
    ma, _ = _moments(rhoa, n)
    z = np.abs(ma - m0) / sd0 + ndtri(a)
    return _power(ndtr(z))


def stplan_correlation_two_sample_power(
    correlation1: ArrayLike,
    correlation2: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Current STPLAN two-sample bivariate-normal correlation approximation."""
    a = _alpha(alpha, sides)
    rho1, rho2, first, second = _inputs(
        (correlation1, "correlation1"),
        (correlation2, "correlation2"),
        (n1, "n1"),
        (n2, "n2"),
    )
    if (
        np.any(np.abs(rho1) >= 1)
        or np.any(np.abs(rho2) >= 1)
        or np.any(first < 4)
        or np.any(second < 4)
    ):
        raise ValueError("correlations must be strictly between -1 and 1; sample sizes >= 4")
    m1, sd1 = _moments(rho1, first)
    m2, sd2 = _moments(rho2, second)
    z = np.abs(m2 - m1) / np.hypot(sd1, sd2) + ndtri(a)
    return _power(ndtr(z))
