"""Bounded, reflected continued-fraction beta integrals."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta_asymptotic import basym
from .cdflib_beta_factors import _deviation, _positive_parts, _scaled
from .cdflib_beta_series import _product_bound, bpser
from .dcdflib_beta import cumbet


def _fraction(
    a: FloatArray,
    b: FloatArray,
    x: FloatArray,
    y: FloatArray,
    lam: FloatArray,
    eps: FloatArray,
) -> FloatArray:
    c, c1 = 1 + lam, 1 + 1 / a
    r = c1 / c
    an, bn = np.zeros(a.shape), r.copy()
    values = np.full(a.shape, np.nan)
    active = np.ones(a.shape, dtype=bool)
    for n in range(1, 513):
        index = np.flatnonzero(active)
        aa, bb, xx, yy = a[index], b[index], x[index], y[index]
        s = aa + 2 * n - 1
        p = 1 + (n - 1) / aa
        w = n * (bb * xx - n * xx)
        e = (1 + n / aa) / (c1[index] + 2 * n / aa)
        beta = n + w / s + e * (c[index] + n * (1 + yy))
        # Regroup x and b/a before forming the potentially overflowing b/a.
        alpha = p * (p * xx + (bb * xx) / aa) * (aa / s) ** 2 * w
        ratio = alpha / beta
        denominator = 1 + ratio * bn[index]
        next_r = (r[index] + ratio * an[index]) / denominator
        valid = np.isfinite(next_r) & (next_r > 0) & np.isfinite(beta) & (beta != 0)
        done = valid & (np.abs(next_r - r[index]) <= eps[index] * next_r)
        values[index[done]] = next_r[done]
        continuing = valid & ~done
        chosen = index[continuing]
        an[chosen] = (r[chosen] / beta[continuing]) / denominator[continuing]
        bn[chosen] = (1 / beta[continuing]) / denominator[continuing]
        r[chosen] = next_r[continuing]
        active[index[~continuing]] = False
        if not np.any(active):
            break
    prefactor, exponent, divisor = _positive_parts(a, b, x, y, np.zeros(a.shape))
    good = np.isfinite(values)
    result = np.full(a.shape, np.nan)
    result[good] = _scaled(prefactor[good], exponent[good] + np.log(values[good]), divisor[good])
    fallback = ~np.isfinite(result) | (result < 0) | (result > 1)
    if np.any(fallback):
        result[fallback] = cumbet(x[fallback], a[fallback], b[fallback], cx=y[fallback])[0]
    return result


def bfrac(
    a: ArrayLike,
    b: ArrayLike,
    x: ArrayLike | None,
    y: ArrayLike | None = None,
    eps: ArrayLike = 5e-15,
) -> FloatArray:
    """Compute I_x(a,b) for a,b>1, preserving either complementary coordinate.

    The source's redundant lambda is computed from the coordinate pair. Inputs
    broadcast; invalid domains raise ValueError and results are owned/immutable.
    """
    xx, yy = _pair(x, y, "x/y")
    aa, bb, xx, yy, ee = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), xx, yy, finite(eps, "eps")
    )
    shape = aa.shape
    aa, bb, xx, yy, ee = (v.ravel() for v in (aa, bb, xx, yy, ee))
    if np.any((aa <= 1) | (bb <= 1) | (ee <= 0)):
        raise ValueError("bfrac requires a,b>1 and positive eps")
    result = np.where(xx == 0, 0.0, 1.0)
    midpoint = (aa == bb) & (xx == yy)
    result[midpoint] = 0.5
    active = np.flatnonzero((xx > 0) & (yy > 0) & ~midpoint)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for start in range(0, active.size, 2048):
            index = active[start : start + 2048]
            a0, b0, x0, y0 = (v[index] for v in (aa, bb, xx, yy))
            lam = _deviation(a0, b0, x0, y0)
            swap = lam < 0
            a0, b0, x0, y0 = (
                np.where(swap, b0, a0),
                np.where(swap, a0, b0),
                np.where(swap, y0, x0),
                np.where(swap, x0, y0),
            )
            lam = np.abs(lam)
            tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(ee[index], 5e-15))
            series = ~_product_bound(np.full(a0.shape, 0.7), b0, x0)
            central = ~series & (np.minimum(a0, b0) > 100) & (lam <= 0.03 * np.minimum(a0, b0))
            value = np.empty(a0.shape)
            if np.any(series):
                value[series] = bpser(a0[series], b0[series], x0[series], tolerance[series])
            if np.any(central):
                value[central] = basym(a0[central], b0[central], lam[central], tolerance[central])
            other = ~(central | series)
            if np.any(other):
                value[other] = _fraction(
                    a0[other], b0[other], x0[other], y0[other], lam[other], tolerance[other]
                )
            result[index] = np.where(swap, 1 - value, value)
    if np.any(~np.isfinite(result) | (result < 0) | (result > 1)):
        raise ArithmeticError("beta continued-fraction evaluation failed")
    return _freeze(result.reshape(shape))
