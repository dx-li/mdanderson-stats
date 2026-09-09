"""Stable beta logarithms and continuous binomial coefficients from CDFLIB F95."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import zeta

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_gamma_ratios import _positive_ratio, algdiv, bcorr
from .cdflib_gamma_support import _local_log_gamma, _positive_log_gamma, psi

_SERIES = tuple(float((-1) ** j * zeta(j, 1) / j) for j in range(2, 41))


def _sum_log_gamma(a: FloatArray, b: FloatArray) -> FloatArray:
    d = (a - 1) + (b - 1)
    near = (d >= -0.2) & (d <= 1.25)
    result = np.empty(a.shape)
    result[near] = _local_log_gamma(d[near]) + np.log1p(d[near])
    result[~near] = _positive_log_gamma(a[~near] + b[~near])
    return result


def _beta_log(x: FloatArray, y: FloatArray) -> FloatArray:
    a, b = np.minimum(x, y), np.maximum(x, y)
    result = np.empty(a.shape)
    unit = (a == 1) | (b == 1)
    result[unit] = -np.log(np.where(a[unit] == 1, b[unit], a[unit]))
    near_unit = (a >= 0.75) & (b <= 1.25) & ~unit
    offset = np.zeros(a.shape)
    offset[near_unit] = (a[near_unit] - 1) + (b[near_unit] - 1)
    near_unit &= np.abs(offset) <= 0.25
    result[near_unit] = -_small_binomial(
        a[near_unit] - 1, b[near_unit] - 1, offset[near_unit]
    ) - np.log1p(offset[near_unit])
    small = (b < 8) & ~(unit | near_unit)
    result[small] = (
        _positive_log_gamma(a[small])
        + _positive_log_gamma(b[small])
        - _sum_log_gamma(a[small], b[small])
    )
    mixed = (a < 8) & (b >= 8) & ~unit
    result[mixed] = _positive_log_gamma(a[mixed]) + _positive_ratio(a[mixed], b[mixed])
    large = (a >= 8) & ~unit
    aa, bb = a[large], b[large]
    ratio = aa / bb
    u = -(aa - 0.5) * np.log(ratio / (1 + ratio))
    v = bb * np.log1p(ratio)
    result[large] = ((-0.5 * np.log(bb) + 0.9189385332046727 + bcorr(aa, bb)) - u) - v
    return result


def betaln(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Compute log B(a,b) for positive finite shapes, with broadcasting."""
    x, y = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    if np.any((x <= 0) | (y <= 0)):
        raise ValueError("beta shapes must be positive")
    with np.errstate(over="ignore", under="ignore"):
        result = _beta_log(x, y)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("beta logarithm exceeds the finite output range")
    return _freeze(result)


def log_beta(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Renamed F95 interface to the positive-shape beta logarithm."""
    return betaln(a, b)


def _small_binomial(k: FloatArray, q: FloatArray, n: FloatArray) -> FloatArray:
    # logGamma(1+n)-logGamma(1+k)-logGamma(1+q), n=k+q.
    # Factor k*q from every degree to eliminate the cancelling linear term.
    # D_j=(n**j-k**j-q**j)/(k*q) obeys D_j=n*D_(j-1)+k**(j-2)+q**(j-2).
    divided = np.full(k.shape, 2.0)
    kp, qp = np.ones(k.shape), np.ones(k.shape)
    series = np.full(k.shape, 2 * _SERIES[0])
    for coefficient in _SERIES[1:]:
        kp *= k
        qp *= q
        divided = n * divided + kp + qp
        series += coefficient * divided
    large = np.where(np.abs(k) >= np.abs(q), k, q)
    small = np.where(np.abs(k) >= np.abs(q), q, k)
    return (large * series) * small


def _small_endpoint(m: FloatArray, n: FloatArray) -> FloatArray:
    n1 = n + 1
    result = np.empty(m.shape)
    tiny = np.abs(m) < 1e-18 * np.minimum(n1, 1)
    result[tiny] = m[tiny] * (psi(n1[tiny]) + 0.5772156649015329)
    m, n1 = m[~tiny], n1[~tiny]
    base = n1 - m
    shift = np.maximum(0, np.ceil(8 - base)).astype(np.int64)
    value = -algdiv(m, base + shift)
    if shift.size:
        for j in range(int(np.max(shift))):
            active = shift > j
            denominator = n1[active] + j
            numerator = base[active] + j
            fraction = -m[active] / denominator
            close = np.abs(fraction) <= 0.5
            term = np.empty(fraction.shape)
            term[close] = np.log1p(fraction[close])
            term[~close] = np.log(numerator[~close]) - np.log(denominator[~close])
            value[active] += term
    local = (m >= -0.2) & (m <= 1.25)
    correction = np.empty(m.shape)
    correction[local] = _local_log_gamma(m[local])
    correction[~local] = _positive_log_gamma(1 + m[~local])
    result[~tiny] = value - correction
    return result


def log_bicoef(k: ArrayLike, n: ArrayLike) -> FloatArray:
    """Compute log[Gamma(n+1)/(Gamma(k+1)*Gamma(n-k+1))] for real inputs.

    The positive-shape domain is n>-1 and -1<k<n+1. Fractional and negative
    values are supported. Arguments broadcast; this is not an integer-only API.
    """
    count, total = np.broadcast_arrays(finite(k, "k"), finite(n, "n"))
    if np.any((count <= -1) | (total <= -1)):
        raise ValueError("n and k must exceed -1")
    # Preserve n when k=1 and preserve n+1 beside n=-1. A naive n-k+1
    # can round a strictly positive shape to zero before the final addition.
    right = np.empty(count.shape)
    negative = total < 0
    around_one = ~negative & (count >= 0.5) & (count <= 2)
    rest = ~(negative | around_one)
    right[negative] = (total[negative] + 1) - count[negative]
    right[around_one] = total[around_one] - (count[around_one] - 1)
    right[rest] = (total[rest] - count[rest]) + 1
    if np.any(right <= 0):
        raise ValueError("k must be less than n+1")
    result = np.empty(count.shape)
    endpoint = (count == 0) | (count == total)
    result[endpoint] = 0
    one = (count == 1) & ~endpoint
    result[one] = np.log(total[one])
    q = total - count
    small = (
        (np.abs(total) <= 0.25) & (np.abs(count) <= 0.25) & (np.abs(q) <= 0.25) & ~endpoint & ~one
    )
    with np.errstate(over="ignore", under="ignore"):
        result[small] = _small_binomial(count[small], q[small], total[small])
        m = np.where(np.abs(count) <= np.abs(q), count, q)
        close = (np.abs(m) <= 0.25) & ~(endpoint | one | small)
        result[close] = _small_endpoint(m[close], total[close])
        regular = ~(endpoint | one | small | close)
        result[regular] = -_beta_log(count[regular] + 1, right[regular]) - np.log1p(total[regular])
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("binomial logarithm exceeds the finite output range")
    return _freeze(result)
