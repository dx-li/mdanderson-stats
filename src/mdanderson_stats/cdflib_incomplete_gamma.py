"""Incomplete-gamma support with explicit endpoints and stable small tails."""

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import exp1

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_error_exponential import _integer
from .cdflib_gamma import _tails as _compiled_tails
from .cdflib_gamma_factor import rcomp
from .cdflib_gamma_ratios import _delta
from .cdflib_gamma_support import _local_log_gamma, _positive_log_gamma
from .dcdflib_gamma import _log_gamma_ratio


def _small_tails(a: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Lower-integral expansion for 0<x<=1.1, preserving small complements."""
    ratio = np.empty(a.shape)
    tiny, local, large = a < 1e-5, (a >= 1e-5) & (a <= 1.25), a >= 8
    ratio[tiny] = _log_gamma_ratio(a[tiny])
    ratio[local] = _local_log_gamma(a[local]) / a[local]
    other = ~(tiny | local | large)
    ratio[other] = _positive_log_gamma(1 + a[other]) / a[other]
    aa = a[large]
    ratio[large] = np.log(aa) - 1 + (0.5 * np.log(aa) + 0.9189385332046727 + _delta(1 / aa)) / aa
    # P=x**a/Gamma(1+a)*(1+a*H), H=sum (-x)**n/(n!*(a+n)).
    # A bounded series suffices since x<=1.1; retain the factor a until last.
    term = -x
    h = term / (a + 1)
    for n in range(2, 41):
        term *= -x / n
        h += term / (a + n)
    v = a * h
    divided = np.ones(a.shape)
    nonzero = v != 0
    divided[nonzero] = np.log1p(v[nonzero]) / v[nonzero]
    logp = a * ((np.log(x) - ratio) + h * divided)
    return np.exp(logp), -np.expm1(logp)


def _upper_ratio(a: FloatArray, x: FloatArray, eps: FloatArray) -> FloatArray:
    """Q/r continued fraction, x>=1.1 and a<=1 or an upper-tail recovery."""
    b = (x - a) + 1
    c = np.full(a.shape, 1e300)
    d = 1 / b
    value = d.copy()
    active = np.ones(a.shape, dtype=bool)
    tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(eps, 1e-14))
    for n in range(1, 1001):
        if not np.any(active):
            return value
        an = n * (a[active] - n)
        b[active] += 2
        den = b[active] + an * d[active]
        num = b[active] + an / c[active]
        den = np.where(np.abs(den) < 1e-300, np.copysign(1e-300, den), den)
        num = np.where(np.abs(num) < 1e-300, np.copysign(1e-300, num), num)
        d[active], c[active] = 1 / den, num
        change = d[active] * num
        value[active] *= change
        active[active] = np.abs(change - 1) > tolerance[active]
    raise ArithmeticError("incomplete gamma continued fraction did not converge")


def _lower_ratio(a: FloatArray, x: FloatArray) -> FloatArray:
    """Endpoint expansion of P/r, used only for subnormal lower tails."""
    d = a - x
    inverse = 1 / d
    curvature = (x / d) / d
    if np.any(curvature > 0.002):
        raise ArithmeticError("lower gamma tail is outside the endpoint expansion region")
    # P/r = integral exp(-a*t+x*(1-exp(-t))) dt, t from 0 to infinity.
    # Expand the nonlinear exponential after scaling t=u/d and integrate
    # each monomial against exp(-u). Coefficients already include factorials.
    terms = [np.ones(a.shape), np.zeros(a.shape)]
    value = terms[0].copy()
    last = np.zeros(a.shape)
    for n in range(2, 33):
        term = np.zeros(a.shape)
        power = curvature.copy()
        for k in range(2, n + 1):
            term += (-1 if k % 2 == 0 else 1) * math.comb(n - 1, k - 1) * power * terms[n - k]
            power *= inverse
        terms.append(term)
        value += term
        if n >= 30:
            last += np.abs(term)
    if np.any(last > 1e-15 * np.abs(value)):
        raise ArithmeticError("lower gamma endpoint expansion did not converge")
    return value / d


def _tails(a: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = np.zeros(a.shape), np.ones(a.shape)
    zero_shape = (a == 0) & (x > 0)
    p[zero_shape], q[zero_shape] = 1, 0
    positive = (a > 0) & (x > 0)
    small = positive & (x <= 1.1)
    p[small], q[small] = _small_tails(a[small], x[small])
    tiny_shape = positive & ~small & (a < 1e-16)
    q[tiny_shape] = a[tiny_shape] * exp1(x[tiny_shape])
    p[tiny_shape] = 1 - q[tiny_shape]
    regular = positive & ~(small | tiny_shape)
    p[regular], q[regular] = _compiled_tails(a[regular], x[regular])
    recover = regular & (q < np.finfo(float).tiny) & (x > a)
    if np.any(recover):
        aa, xx = a[recover], x[recover]
        r = rcomp(aa, xx)
        values = np.zeros(aa.shape)
        nonzero = r > 0
        values[nonzero] = r[nonzero] * _upper_ratio(
            aa[nonzero], xx[nonzero], np.full(np.count_nonzero(nonzero), 1e-14)
        )
        q[recover], p[recover] = values, 1 - values
    recover = regular & (p < np.finfo(float).tiny) & (x < a)
    if np.any(recover):
        aa, xx = a[recover], x[recover]
        r = rcomp(aa, xx)
        values = np.zeros(aa.shape)
        nonzero = r > 0
        values[nonzero] = r[nonzero] * _lower_ratio(aa[nonzero], xx[nonzero])
        p[recover], q[recover] = values, 1 - values
    return p, q


def _finish(p: FloatArray, q: FloatArray) -> tuple[FloatArray, FloatArray]:
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("incomplete gamma evaluation produced invalid probabilities")
    return _freeze(p), _freeze(q)


def gratio(a: ArrayLike, x: ArrayLike, ind: ArrayLike = 0) -> tuple[FloatArray, FloatArray]:
    """Return P(a,x),Q(a,x) for finite nonnegative a,x, excluding (0,0).

    Signed int32 accuracy selectors broadcast; all modes use full float64
    accuracy. Invalid domains raise ValueError instead of native sentinels.
    """
    aa, xx, _ = np.broadcast_arrays(finite(a, "a"), finite(x, "x"), _integer(ind, "ind"))
    if np.any((aa < 0) | (xx < 0) | ((aa == 0) & (xx == 0))):
        raise ValueError("a and x must be nonnegative and not both zero")
    with np.errstate(over="ignore", under="ignore"):
        return _finish(*_tails(aa, xx))


def grat1(
    a: ArrayLike, x: ArrayLike, r: ArrayLike, eps: ArrayLike = 5e-15
) -> tuple[FloatArray, FloatArray]:
    """Return small-shape gamma tails for 0<=a<=1, x>=0 and positive eps.

    The supplied nonnegative scaling factor r is consumed in the continued
    fraction branch (a!=0.5,x>=1.1), as in F95. Other branches ignore it.
    All inputs broadcast. At (a,x)=(0,0), preserve the native (0,1) return.
    """
    aa, xx, rr, tolerance = np.broadcast_arrays(
        finite(a, "a"), finite(x, "x"), finite(r, "r"), finite(eps, "eps")
    )
    if np.any((aa < 0) | (aa > 1) | (xx < 0) | (rr < 0) | (tolerance <= 0)):
        raise ValueError("require 0<=a<=1, x>=0, r>=0 and eps>0")
    with np.errstate(over="ignore", under="ignore"):
        fraction = (aa > 0) & (aa != 0.5) & (xx >= 1.1)
        p, q = np.empty(aa.shape), np.empty(aa.shape)
        p[~fraction], q[~fraction] = _tails(aa[~fraction], xx[~fraction])
        q[fraction] = rr[fraction] * _upper_ratio(aa[fraction], xx[fraction], tolerance[fraction])
        p[fraction] = 1 - q[fraction]
        return _finish(p, q)
