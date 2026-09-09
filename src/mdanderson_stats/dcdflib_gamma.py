"""Legacy DCDFLIB gamma contracts with explicit rate and scaled-tail repairs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import exp1, gammainccinv, gammaincinv, gammaln

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray, finite
from .cdflib_gamma import _tails as _unit_tails
from .dcdflib_f import _positive


def _log_gamma_ratio(a: FloatArray) -> FloatArray:
    """logGamma(1+a)/a without rounding 1+a to one for tiny shapes."""
    small = a < 1e-5
    result = np.empty(a.shape)
    x = a[small]
    # Taylor coefficients: -EulerGamma, zeta(2)/2, -zeta(3)/3.
    result[small] = -0.5772156649015329 + x * (0.8224670334241132 - x * 0.4006856343865314)
    result[~small] = gammaln(1 + a[~small]) / a[~small]
    return result


def _tails(x: FloatArray, a: FloatArray, rate: FloatArray) -> tuple[FloatArray, FloatArray]:
    with np.errstate(over="ignore"):
        z = x * rate
    p, q = np.zeros(x.shape), np.ones(x.shape)
    positive = x > 0
    tiny = positive & (z < np.finfo(float).tiny)
    small_shape = positive & ~tiny & (a < 1e-12)
    regular = positive & ~tiny & ~small_shape
    # Q(a,z) = a*E1(z)*(1+O(a*(1+abs(log(z))))). In this branch
    # z is normal or larger and a<1e-12, so the relative error is below 1e-9.
    q[small_shape] = a[small_shape] * exp1(z[small_shape])
    p[small_shape] = 1 - q[small_shape]
    p[regular], q[regular] = _unit_tails(a[regular], z[regular])
    if np.any(tiny):
        logz = np.log(x[tiny]) + np.log(rate[tiny])
        with np.errstate(over="ignore"):
            logp = a[tiny] * (logz - _log_gamma_ratio(a[tiny]))
        p[tiny], q[tiny] = np.exp(logp), -np.expm1(logp)
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("legacy gamma tail evaluation failed")
    return p, q


def _exp1_log_inverse(target: FloatArray, initial: FloatArray) -> FloatArray:
    low, high = initial.copy(), np.full(initial.shape, np.log(750.0))
    best, error = low.copy(), np.full(low.shape, np.inf)
    for _ in range(64):
        middle = (low + high) / 2
        value = exp1(np.exp(middle))
        residual = np.abs(value - target)
        best = np.where(residual < error, middle, best)
        error = np.minimum(error, residual)
        move_low = value > target
        low, high = np.where(move_low, middle, low), np.where(move_low, high, middle)
    return best


def _scaled_quantile(
    p: FloatArray, q: FloatArray, a: FloatArray, divisor: FloatArray
) -> FloatArray:
    lower = p <= q
    small_shape = a < 1e-12
    z = np.zeros(p.shape)
    regular = ~small_shape
    left, right = regular & lower, regular & ~lower
    z[left] = gammaincinv(a[left], p[left])
    z[right] = gammainccinv(a[right], q[right])
    if np.any(~np.isfinite(z) | (z < 0)):
        raise ArithmeticError("legacy gamma inverse kernel failed")
    with np.errstate(over="ignore"):
        result = np.array(z / divisor, copy=True)
    small = (z < np.finfo(float).tiny) & (p > 0)
    if np.any(small):
        with np.errstate(divide="ignore", over="ignore"):
            logp = np.where(lower[small], np.log(p[small]), np.log1p(-q[small]))
            logz = logp / a[small] + _log_gamma_ratio(a[small])
        if np.any(~np.isfinite(logz) & (logz != -np.inf)):
            raise ArithmeticError("gamma logarithmic inverse kernel failed")
        normal = logz >= np.log(np.finfo(float).tiny)
        if np.any(normal & ~small_shape[small]):
            raise ArithmeticError("gamma inverse kernel failed outside its subnormal regime")
        if np.any(normal):
            logz[normal] = _exp1_log_inverse(q[small][normal] / a[small][normal], logz[normal])
        with np.errstate(over="ignore"):
            result[small] = np.exp(logz - np.log(divisor[small]))
    return np.where(p == 0, 0.0, result)


@dataclass(frozen=True)
class DCDFLIBGamma:
    """Computed group and owned immutable p/q, x, shape and rate arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    x: FloatArray
    shape: FloatArray
    rate: FloatArray


def cdfgam(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    x: ArrayLike | None = None,
    shape: ArrayLike | None = None,
    rate: ArrayLike | None = None,
) -> DCDFLIBGamma:
    """Compute 1=p/q, 2=x, 3=shape or 4=rate for the legacy C/F77 interface.

    The source SCALE is a rate, multiplying x. Input rate defaults to one.
    Only computed shape has search bounds [1e-100,1e100]; input shape/rate
    and computed rate are positive finite, x and computed x nonnegative finite.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (x,), 3: (shape,), 4: (rate,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if which != 2 and x is None:
        raise ValueError("x is required")
    xx = finite(0.0 if x is None else x, "x")
    if np.any(xx < 0):
        raise ValueError("x must be nonnegative")
    aa = _positive(shape, "shape") if which != 3 else np.asarray(1.0)
    rr = _positive(1.0 if rate is None else rate, "rate")
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, xx, aa, rr = np.broadcast_arrays(pp, qq, xx, aa, rr)
    if which == 1:
        pp, qq = _tails(xx, aa, rr)
    else:
        if np.any(qq <= 0):
            raise ValueError("legacy gamma inversion requires positive q")
        if which in (3, 4) and np.any((pp <= 0) | (xx <= 0)):
            raise ValueError("shape/rate inversion requires positive x and p/q")
        if which == 2:
            xx = _scaled_quantile(pp, qq, aa, rr)
            if np.any(~np.isfinite(xx) | (xx < 0) | ((xx == 0) & (pp > 0))):
                raise ValueError("gamma x is not representable as a nonnegative finite float")
        elif which == 4:
            rr = _scaled_quantile(pp, qq, aa, xx)
            if np.any(~np.isfinite(rr) | (rr <= 0)):
                raise ValueError("gamma rate is not representable as a positive finite float")
        else:
            fixed, rates, lower = xx.ravel(), rr.ravel(), (pp <= qq).ravel()

            def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(fixed[indices], value, rates[indices])
                return np.where(lower[indices], lp, uq)

            with np.errstate(over="ignore"):
                z = xx * rr
            aa = _invert_positive(
                np.minimum(pp, qq),
                np.full(pp.shape, 1e-100),
                np.full(pp.shape, 1e100),
                evaluate,
                initial=np.where(z >= 1, z, 5.0),
                unbracketed_message="gamma shape solution lies outside [1e-100,1e100]",
            )
        lp, uq = _tails(xx, aa, rr)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy gamma inversion failed forward verification")
    return DCDFLIBGamma(int(which), *map(_freeze, (pp, qq, xx, aa, rr)))


def cumgam(x: ArrayLike, shape: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired unit-rate gamma tails; native x is the already-scaled coordinate."""
    result = cdfgam(x=x, shape=shape)
    return result.p, result.q
