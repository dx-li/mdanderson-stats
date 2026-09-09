"""Scaled beta factors with stable centers and complete exponential scaling."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, gammasgn

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta_support import _small_binomial
from .cdflib_elementary import _log_remainder
from .cdflib_error_exponential import _integer
from .cdflib_gamma_ratios import _positive_ratio, algdiv, bcorr
from .cdflib_gamma_support import _local_log_gamma, _positive_log_gamma


def _log_gamma_one(a: FloatArray) -> FloatArray:
    local = a <= 1.25
    result = np.empty(a.shape)
    result[local] = _local_log_gamma(a[local])
    result[~local] = _positive_log_gamma(1 + a[~local])
    return result


def _product_parts(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    # Dekker product on normalized mantissas avoids overflowing the splitter.
    ma, ea = np.frexp(a)
    mb, eb = np.frexp(b)
    ca, cb = 134217729 * ma, 134217729 * mb
    ah, bh = ca - (ca - ma), cb - (cb - mb)
    al, bl = ma - ah, mb - bh
    product = ma * mb
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    return np.ldexp(product, ea + eb), np.ldexp(error, ea + eb)


def _deviation(a: FloatArray, b: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    first, first_error = _product_parts(a, y)
    second, second_error = _product_parts(b, x)
    # The larger coordinate is rounded; account for the exact complement of
    # the smaller supplied coordinate when evaluating a*y-b*x near its zero.
    correction = np.where(x <= y, a * ((1 - y) - x), -b * ((1 - x) - y))
    return (first - second) + ((first_error - second_error) + correction)


def _positive_parts(
    a: FloatArray, b: FloatArray, x: FloatArray, y: FloatArray, mu: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    swap = (a > b) | ((a == b) & (x > y))
    a, b, x, y = (
        np.where(swap, b, a),
        np.where(swap, a, b),
        np.where(swap, y, x),
        np.where(swap, x, y),
    )
    lnx = np.where(x <= y, np.log(x), np.log1p(-y))
    lny = np.where(y < x, np.log(y), np.log1p(-x))
    exponent = mu + a * lnx + b * lny
    prefactor, divisor = np.ones(a.shape), np.ones(a.shape)
    unit = (a == 1) | (b == 1)
    prefactor[unit] = np.where(a[unit] == 1, b[unit], a[unit])
    small = (b < 8) & ~unit
    if np.any(small):
        aa, bb = a[small], b[small]
        prefactor[small], divisor[small] = aa, 1 + aa / bb
        s = aa + bb
        local = s <= 0.25
        correction = np.empty(s.shape)
        if np.any(local):
            correction[local] = _small_binomial(aa[local], bb[local], s[local])
        correction[~local] = (
            _log_gamma_one(s[~local]) - _log_gamma_one(aa[~local]) - _log_gamma_one(bb[~local])
        )
        exponent[small] += correction
    mixed = (a < 8) & (b >= 8) & ~unit
    if np.any(mixed):
        prefactor[mixed] = a[mixed]
        exponent[mixed] -= _log_gamma_one(a[mixed]) + _positive_ratio(a[mixed], b[mixed])
    large = a >= 8
    if np.any(large):
        aa, bb, xx, yy = a[large], b[large], x[large], y[large]
        lx, ly = lnx[large], lny[large]
        h = aa / bb
        x0, y0 = h / (1 + h), 1 / (1 + h)
        offset = _deviation(aa, bb, xx, yy)
        ea, eb = -offset / aa, offset / bb
        u, v = np.empty(ea.shape), np.empty(eb.shape)
        near_a, near_b = np.abs(ea) <= 0.6, np.abs(eb) <= 0.6
        u[near_a], v[near_b] = _log_remainder(ea[near_a]), _log_remainder(eb[near_b])
        u[~near_a] = ea[~near_a] - (lx[~near_a] - np.log(x0[~near_a]))
        v[~near_b] = eb[~near_b] - (ly[~near_b] - np.log(y0[~near_b]))
        exponent[large] = (
            mu[large]
            + 0.5 * (np.log(aa) - np.log1p(h))
            - 0.9189385332046727
            - bcorr(aa, bb)
            - aa * u
            - bb * v
        )
    return prefactor, exponent, divisor


def _small_coordinate_log_cdf(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Leading beta integral for x<=1e-300 and b*x<=1e-16.

    The relative integral-series correction is bounded by order max(x,b*x).
    Retain log(P) so both P and a small complementary Q remain recoverable.
    """
    result = np.empty(a.shape)
    small = a <= 1
    if np.any(small):
        aa, shifted = a[small], b[small].copy()
        correction = np.zeros(aa.shape)
        # Shift b into the stable gamma-ratio domain. Gamma recurrences
        # avoid subtracting log(a) from log(Beta(a,b)) for tiny a.
        for _ in range(8):
            active = shifted < 8
            ratio = aa[active] / shifted[active]
            logarithm = np.log1p(ratio)
            overflow = np.isinf(ratio)
            logarithm[overflow] = np.log(aa[active][overflow]) - np.log(shifted[active][overflow])
            correction[active] -= logarithm
            shifted[active] += 1
        result[small] = (
            aa * np.log(x[small]) - _local_log_gamma(aa) - _positive_ratio(aa, shifted) + correction
        )
    if np.any(~small):
        aa, bb, xx = a[~small], b[~small], x[~small]
        prefactor, exponent, divisor = _positive_parts(aa, bb, xx, 1 - xx, np.zeros(aa.shape))
        result[~small] = (
            exponent - np.log(aa) + np.log(prefactor) - np.log(divisor) - bb * np.log1p(-xx)
        )
    return result


def _scaled(prefactor: FloatArray, exponent: FloatArray, divisor: FloatArray) -> FloatArray:
    exponent = exponent - np.log(divisor)
    direct = np.abs(exponent) <= 700
    split = (np.abs(exponent) <= 1400) & ~direct
    result = np.empty(prefactor.shape)
    result[direct] = prefactor[direct] * np.exp(exponent[direct])
    half = np.exp(exponent[split] / 2)
    result[split] = (prefactor[split] * half) * half
    rest = ~(direct | split)
    result[rest] = np.exp(np.log(prefactor[rest]) + exponent[rest])
    return result


def _log_abs_gamma(a: FloatArray) -> FloatArray:
    local = np.abs(a) <= 0.2
    result = np.empty(a.shape)
    result[local] = _local_log_gamma(a[local]) - np.log(np.abs(a[local]))
    result[~local] = gammaln(a[~local])
    return result


def _sum_parts(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    total = a + b
    recovered = total - a
    error = (a - (total - recovered)) + (b - recovered)
    return total, error


def _log_sum_gamma(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    total, error = _sum_parts(a, b)
    negative = total <= 0
    logarithm, sign = np.empty(total.shape), np.ones(total.shape)
    logarithm[~negative] = _positive_log_gamma(total[~negative])
    nearest = np.rint(total[negative])
    offset = (total[negative] - nearest) + error[negative]
    turn = np.rint(offset)
    nearest, offset = nearest + turn, offset - turn
    # Reflection retains the true distance to a pole even when the rounded
    # sum lands exactly on that pole. The individual shapes were validated.
    logarithm[negative] = (
        np.log(np.pi)
        - np.log(np.abs(np.sin(np.pi * offset)))
        - _positive_log_gamma((1 - total[negative]) - error[negative])
    )
    sign[negative] = np.where(np.remainder(nearest, 2) == 0, 1, -1) * np.sign(offset)
    return logarithm, sign


def brcmp1(
    mu: ArrayLike, a: ArrayLike, b: ArrayLike, x: ArrayLike | None, y: ArrayLike | None = None
) -> FloatArray:
    """Compute exp(mu)*x**a*y**b/Beta(a,b), preserving the smaller coordinate.

    Finite inputs broadcast; mu is int32 and x,y are complementary probabilities.
    Positive shapes and the Gamma-defined real domain are supported. Gamma poles
    are rejected, except a zero shape with a positive companion returns zero.
    Endpoint coordinates return zero, as in F95. Output overflow is explicit.
    """
    xx, yy = _pair(x, y, "x,y")
    aa, bb, xx, yy, shift = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), xx, yy, _integer(mu, "mu")
    )
    zero = ((aa == 0) & (bb > 0)) | ((bb == 0) & (aa > 0))
    nonpositive = ((aa <= 0) | (bb <= 0)) & ~zero
    if np.any((aa[nonpositive] <= 0) & (aa[nonpositive] == np.floor(aa[nonpositive]))) or np.any(
        (bb[nonpositive] <= 0) & (bb[nonpositive] == np.floor(bb[nonpositive]))
    ):
        raise ValueError("a, b and a+b must avoid Gamma poles")
    total, error = _sum_parts(aa[nonpositive], bb[nonpositive])
    if np.any((total <= 0) & (total == np.floor(total)) & (error == np.floor(error))):
        raise ValueError("a, b and a+b must avoid Gamma poles")
    result = np.zeros(aa.shape)
    active = (xx > 0) & (yy > 0) & ~zero
    positive = active & (aa > 0) & (bb > 0)
    negative = active & ~positive
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        result[positive] = _scaled(
            *_positive_parts(
                aa[positive], bb[positive], xx[positive], yy[positive], shift[positive]
            )
        )
        if np.any(negative):
            a0, b0 = np.minimum(aa[negative], bb[negative]), np.maximum(aa[negative], bb[negative])
            s = a0 + b0
            inverse_beta = np.empty(s.shape)
            mixed = (b0 >= 8) & (s > 0)
            inverse_beta[mixed] = -_log_abs_gamma(a0[mixed]) - algdiv(a0[mixed], b0[mixed])
            sum_logarithm, sum_sign = _log_sum_gamma(a0[~mixed], b0[~mixed])
            inverse_beta[~mixed] = (
                sum_logarithm - _log_abs_gamma(a0[~mixed]) - _log_abs_gamma(b0[~mixed])
            )
            nx, ny = xx[negative], yy[negative]
            lnx = np.where(nx <= ny, np.log(nx), np.log1p(-ny))
            lny = np.where(ny < nx, np.log(ny), np.log1p(-nx))
            sign = gammasgn(a0) * gammasgn(b0)
            sign[~mixed] *= sum_sign
            result[negative] = sign * np.exp(
                shift[negative] + aa[negative] * lnx + bb[negative] * lny + inverse_beta
            )
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("scaled beta factor exceeds the finite output range")
    return _freeze(result)


def brcomp(
    a: ArrayLike, b: ArrayLike, x: ArrayLike | None, y: ArrayLike | None = None
) -> FloatArray:
    """Compute x**a*y**b/Beta(a,b); equivalent to brcmp1(0,a,b,x,y)."""
    return brcmp1(0, a, b, x, y)
