"""Gamma/digamma support from CDFLIB's public F95 mathematical module."""

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_elementary import evaluate_polynomial

# Morris's local rational approximations, in ascending coefficient order.
# Original provenance and retained notice: docs/cdflib-gamma-support-reference.md.
_P = (
    0.577215664901533,
    0.844203922187225,
    -0.168860593646662,
    -0.780427615533591,
    -0.402055799310489,
    -0.0673562214325671,
    -0.00271935708322958,
)
_Q = (
    1.0,
    2.88743195473681,
    3.12755088914843,
    1.56875193295039,
    0.361951990101499,
    0.0325038868253937,
    0.000667465618796164,
)
_R = (
    0.422784335098467,
    0.848044614534529,
    0.565221050691933,
    0.156513060486551,
    0.017050248402265,
    0.000497958207639485,
)
_S = (
    1.0,
    1.24313399877507,
    0.548042109832463,
    0.10155218743983,
    0.00713309612391,
    0.000116165475989616,
)


def _local_log_gamma(a: FloatArray) -> FloatArray:
    low = a < 0.6
    result = np.empty(a.shape)
    z = a[low]
    result[low] = -z * (evaluate_polynomial(_P, z) / evaluate_polynomial(_Q, z))
    z = a[~low] - 1
    result[~low] = z * (evaluate_polynomial(_R, z) / evaluate_polynomial(_S, z))
    return result


def _positive_log_gamma(x: FloatArray) -> FloatArray:
    result = np.empty(x.shape)
    small = x <= 0.8
    near = (x > 0.8) & (x <= 2.25)
    result[small] = _local_log_gamma(x[small]) - np.log(x[small])
    result[near] = _local_log_gamma(x[near] - 1)
    result[~(small | near)] = special.gammaln(x[~(small | near)])
    return result


def _finish(result: FloatArray, name: str) -> FloatArray:
    if np.any(~np.isfinite(result)):
        raise ArithmeticError(f"{name} exceeds the finite output range")
    return _freeze(result)


def gamln(a: ArrayLike) -> FloatArray:
    """Evaluate log Gamma(a) for positive finite a, preserving roots at 1 and 2."""
    x = finite(a, "a")
    if np.any(x <= 0):
        raise ValueError("a must be positive")
    return _finish(_positive_log_gamma(x), "log gamma")


def log_gamma(a: ArrayLike) -> FloatArray:
    """Evaluate log Gamma(a) on the positive domain of the F95 interface."""
    return gamln(a)


def _nonpole(a: ArrayLike) -> FloatArray:
    x = finite(a, "a")
    if np.any((x <= 0) & (x == np.floor(x))):
        raise ValueError("nonpositive integer arguments are poles")
    return x


def alngam(x: ArrayLike) -> FloatArray:
    """Evaluate the real log Gamma(x), including intervals with positive Gamma(x).

    Negative nonintegers with negative Gamma(x) have no real logarithm and are
    rejected, as are nonpositive integer poles.
    """
    value = _nonpole(x)
    if np.any(special.gammasgn(value) < 0):
        raise ValueError("gamma must be positive for a real logarithm")
    positive = value > 0
    result = np.empty(value.shape)
    result[positive] = _positive_log_gamma(value[positive])
    result[~positive] = special.gammaln(value[~positive])
    return _finish(result, "log gamma")


def gamln1(a: ArrayLike) -> FloatArray:
    """Compute log Gamma(1+a) for -0.2<=a<=1.25 without losing small a."""
    value = finite(a, "a")
    if np.any((value < -0.2) | (value > 1.25)):
        raise ValueError("a must lie in [-0.2, 1.25]")
    return _finish(_local_log_gamma(value), "log gamma remainder")


def gam1(a: ArrayLike) -> FloatArray:
    """Compute 1/Gamma(1+a)-1 for -0.5<=a<=1.5 without cancellation."""
    value = finite(a, "a")
    if np.any((value < -0.5) | (value > 1.5)):
        raise ValueError("a must lie in [-0.5, 1.5]")
    local = (value >= -0.2) & (value <= 1.25)
    log_value = np.empty(value.shape)
    log_value[local] = _local_log_gamma(value[local])
    log_value[~local] = special.gammaln(1 + value[~local])
    return _finish(np.expm1(-log_value), "reciprocal gamma remainder")


def gamma(a: ArrayLike) -> FloatArray:
    """Evaluate real Gamma(a), retaining signed subnormal negative-argument tails.

    Poles raise ValueError; output overflow raises ArithmeticError. True output
    underflow is permitted and preserves the sign.
    """
    value = _nonpole(a)
    tail = value < -170
    result = np.empty(value.shape)
    with np.errstate(over="ignore", under="ignore"):
        result[~tail] = special.gamma(value[~tail])
        result[tail] = special.gammasgn(value[tail]) * np.exp(special.gammaln(value[tail]))
    return _finish(result, "gamma")


def psi(x: ArrayLike) -> FloatArray:
    """Evaluate digamma for finite nonpole real x, including negative arguments."""
    value = _nonpole(x)
    with np.errstate(over="ignore"):
        result = special.digamma(value)
    return _finish(result, "digamma")
