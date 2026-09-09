"""Stable elementary functions from CDFLIB's public mathematical support."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite


def alnrel(a: ArrayLike) -> FloatArray:
    """Compute log(1+a) for finite a>-1, preserving small arguments."""
    x = finite(a, "a")
    if np.any(x <= -1):
        raise ValueError("a must exceed -1")
    return _freeze(np.log1p(x))


def rexp(x: ArrayLike) -> FloatArray:
    """Compute exp(x)-1; raise if the result exceeds finite binary64."""
    value = finite(x, "x")
    with np.errstate(over="ignore"):
        result = np.expm1(value)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("exp(x)-1 exceeds the finite output range")
    return _freeze(result)


def _log_remainder(x: FloatArray) -> FloatArray:
    small = np.abs(x) <= 0.125
    result = np.empty(x.shape)
    z = x[small]
    # x-log(1+x) = x**2 * sum_{k=2}^infinity (-x)**(k-2)/k.
    # The omitted relative tail after k=40 is < 2*(1/8)**39.
    series = np.full(z.shape, 1 / 40)
    for k in range(39, 1, -1):
        series = (-1.0 if k % 2 else 1.0) / k + z * series
    result[small] = (z * series) * z
    result[~small] = x[~small] - np.log1p(x[~small])
    return result


def rlog1(x: ArrayLike) -> FloatArray:
    """Compute x-log(1+x) for x>-1 without cancellation near zero."""
    value = finite(x, "x")
    if np.any(value <= -1):
        raise ValueError("x must exceed -1")
    return _freeze(_log_remainder(value))


def rlog(x: ArrayLike) -> FloatArray:
    """Compute x-1-log(x) for positive finite x, accurately near one."""
    value = finite(x, "x")
    if np.any(value <= 0):
        raise ValueError("x must be positive")
    near = np.abs(value - 1) <= 0.125
    result = np.empty(value.shape)
    result[near] = _log_remainder(value[near] - 1)
    result[~near] = (value[~near] - 1) - np.log(value[~near])
    return _freeze(result)


def evaluate_polynomial(a: ArrayLike, x: ArrayLike) -> FloatArray:
    """Evaluate a[0]+a[1]*x+... with Horner's rule over a batch of x values.

    Coefficients are a nonempty one-dimensional sequence. Inputs and outputs
    must be finite; an overflowing Horner intermediate raises ArithmeticError.
    """
    coefficients, value = finite(a, "a"), finite(x, "x")
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("a must be a nonempty one-dimensional coefficient sequence")
    result = np.full(value.shape, coefficients[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        for coefficient in coefficients[-2::-1]:
            result = coefficient + result * value
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("polynomial evaluation exceeded the finite output range")
    return _freeze(result)
