"""Stable gamma ratios and Stirling corrections from CDFLIB F95 support."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_gamma_support import _local_log_gamma

# B_(2k)/(2k(2k-1)); ten terms suffice at x>=8. Unlike subtraction
# of log gamma values, this series retains small corrections at large x.
_STIRLING = (
    1 / 12,
    -1 / 360,
    1 / 1260,
    -1 / 1680,
    1 / 1188,
    -691 / 360360,
    1 / 156,
    -3617 / 122400,
    43867 / 244188,
    -174611 / 125400,
)


def _delta(inverse: FloatArray) -> FloatArray:
    square = inverse * inverse
    value = np.full(inverse.shape, _STIRLING[-1])
    for coefficient in _STIRLING[-2::-1]:
        value = coefficient + square * value
    return inverse * value


def bcorr(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Compute delta(a)+delta(b)-delta(a+b) for finite a,b>=8.

    Delta is the correction to Stirling's leading log-gamma expression.
    Arguments broadcast; their sum need not fit in float64.
    """
    x, y = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    if np.any((x < 8) | (y < 8)):
        raise ValueError("a and b must be at least 8")
    small, large = np.minimum(x, y), np.maximum(x, y)
    with np.errstate(under="ignore"):
        inverse_sum = (1 / large) / (1 + small / large)
        result = _delta(1 / small) + _delta(1 / large) - _delta(inverse_sum)
    return _freeze(result)


def _positive_ratio(a: FloatArray, b: FloatArray) -> FloatArray:
    """log Gamma(b)-log Gamma(b+a), with a>=0 and b>=8."""
    r = a / b
    inverse = 1 / b
    quotient = np.ones(r.shape)
    active = r > 0
    quotient[active] = np.log1p(r[active]) / r[active]
    # Reassociate (b+a-.5)*log1p(a/b) so a/b may underflow without
    # discarding the leading contribution a*(1-.5/b).
    u = a * ((1 + r - 0.5 * inverse) * quotient)
    v = a * (np.log(b) - 1)
    # Delta(b)-delta(b+a): avoid subtracting nearly equal corrections.
    # 1-t**n=(1-t)*(1+t+...+t**(n-1)); n is odd and t=1/(1+r).
    t = 1 / (1 + r)
    square = inverse * inverse
    power = np.ones(r.shape)
    geometric = np.ones(r.shape)
    correction = np.full(r.shape, _STIRLING[0])
    for coefficient in _STIRLING[1:]:
        power *= square
        geometric = 1 + t + t * t * geometric
        correction += coefficient * power * geometric
    w = a * (inverse * inverse * t * correction)
    return (w - u) - v


def algdiv(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Compute log Gamma(b)-log Gamma(a+b), with b>=8 and a+b>0.

    Finite arguments broadcast. Negative a is supported on this domain.
    Positive sums may exceed float64; output overflow raises ArithmeticError.
    """
    x, y = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    if np.any(y < 8):
        raise ValueError("b must be at least 8")
    negative = x < 0
    base = y[negative] + x[negative]
    if np.any(base <= 0):
        raise ValueError("a+b must be positive")
    result = np.empty(x.shape)
    with np.errstate(over="ignore", under="ignore"):
        result[~negative] = _positive_ratio(x[~negative], y[~negative])
        if np.any(negative):
            nx, ny = x[negative], y[negative]
            step = -nx
            # Shift both arguments to the region where the Stirling remainder
            # is accurate. At most eight recurrence terms are needed.
            shift = np.maximum(0, np.ceil(8 - base)).astype(np.int64)
            value = -_positive_ratio(step, base + shift)
            for j in range(int(np.max(shift))):
                active = shift > j
                numerator, denominator = base[active] + j, ny[active] + j
                fraction = nx[active] / denominator
                close = np.abs(fraction) <= 0.5
                term = np.empty(fraction.shape)
                term[close] = np.log1p(fraction[close])
                term[~close] = np.log(numerator[~close]) - np.log(denominator[~close])
                value[active] += term
            result[negative] = value
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("log gamma ratio exceeds the finite output range")
    return _freeze(result)


def gsumln(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Compute log Gamma(a+b) for a,b in [1,2], preserving offsets from two."""
    x, y = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    if np.any((x < 1) | (x > 2) | (y < 1) | (y > 2)):
        raise ValueError("a and b must lie in [1,2]")
    d = (x - 1) + (y - 1)
    low = d <= 1.25
    result = np.empty(d.shape)
    result[low] = _local_log_gamma(d[low]) + np.log1p(d[low])
    result[~low] = _local_log_gamma(d[~low] - 1) + np.log(d[~low]) + np.log1p(d[~low])
    return _freeze(result)
