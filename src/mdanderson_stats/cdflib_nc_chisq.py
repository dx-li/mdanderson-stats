"""CDFLIB90 noncentral chi-square tails and all three parameter inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ncx2

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_chisq import _df
from .cdflib_gamma import _bounded


def _noncentrality(value: ArrayLike | None) -> FloatArray:
    if value is None:
        raise ValueError("pnonc is required")
    result = finite(value, "pnonc")
    if np.any((result < 0) | (result > 1e4)):
        raise ValueError("pnonc must lie in [0,1e4]")
    return result


def _tails(x: FloatArray, df: FloatArray, nc: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = ncx2.cdf(x, df, nc), ncx2.sf(x, df, nc)
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("noncentral chi-square tail evaluation failed")
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


def _verify(p: FloatArray, q: FloatArray, x: FloatArray, df: FloatArray, nc: FloatArray) -> None:
    lp, uq = _tails(x, df, nc)
    target = np.minimum(p, q)
    if np.any(
        np.abs(np.where(p <= q, lp, uq) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
    ):
        raise ArithmeticError("noncentral chi-square inversion failed forward verification")


def _invert_parameter(
    p: FloatArray, q: FloatArray, x: FloatArray, df: FloatArray, nc: FloatArray, *, solve_df: bool
) -> FloatArray:
    lower, target = p <= q, np.minimum(p, q)

    def evaluate(value: FloatArray) -> FloatArray:
        lp, uq = _tails(x, value if solve_df else df, nc if solve_df else value)
        return np.where(lower, lp, uq)

    low = np.full(p.shape, 1e-3 if solve_df else 0.0)
    high = np.full(p.shape, 1e10 if solve_df else 1e4)
    at_low, at_high = evaluate(low), evaluate(high)
    if np.any(at_low == at_high):
        raise ValueError("parameter is numerically unidentified at this x")
    adjusted = target.copy()
    for endpoint in (at_low, at_high):
        adjusted = np.where(
            np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint, endpoint, adjusted
        )
    if np.any((adjusted < np.minimum(at_low, at_high)) | (adjusted > np.maximum(at_low, at_high))):
        raise ValueError("noncentral chi-square parameter solution lies outside its bounds")
    # Both CDFs decrease with the parameter. Log-df and linear noncentrality
    # keep the bounded search efficient without assuming a positive nc root.
    lo, hi = (np.log(low), np.log(high)) if solve_df else (low.copy(), high.copy())
    for _ in range(64):
        middle = (lo + hi) / 2
        value = evaluate(np.exp(middle) if solve_df else middle)
        move_low = np.where(lower, value > adjusted, value < adjusted)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
    result = np.exp((lo + hi) / 2) if solve_df else (lo + hi) / 2
    return np.where(adjusted == at_low, low, np.where(adjusted == at_high, high, result))


@dataclass(frozen=True)
class CDFNoncentralChiSquare:
    """Computed group and five owned immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    x: FloatArray
    df: FloatArray
    pnonc: FloatArray


def cdf_nc_chisq(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    x: ArrayLike | None = None,
    df: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
) -> CDFNoncentralChiSquare:
    """Compute 1=tails, 2=x, 3=df or 4=pnonc; omit the computed group.

    x is in [0,1e100], df in [1e-3,1e10] and pnonc in [0,1e4].
    Noncentrality is the sum of squared normal means, not their sum.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (x,), 3: (df,), 4: (pnonc,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    xx = _bounded(x, "x") if which != 2 else np.asarray(0.0)
    degrees = _df(df) if which != 3 else np.asarray(1.0)
    nc = _noncentrality(pnonc) if which != 4 else np.asarray(0.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, xx, degrees, nc = np.broadcast_arrays(p, q, xx, degrees, nc)
    if which == 1:
        p, q = _tails(xx, degrees, nc)
    elif which == 2:
        if np.any(q <= 0):
            raise ValueError("zero ccum has no finite noncentral chi-square quantile")
        lower = p <= q
        xx = np.empty(p.shape)
        xx[lower] = ncx2.ppf(p[lower], degrees[lower], nc[lower])
        xx[~lower] = ncx2.isf(q[~lower], degrees[~lower], nc[~lower])
        xx = _bounded(xx, "x", computed=True)
        _verify(p, q, xx, degrees, nc)
    else:
        if np.any((p <= 0) | (q <= 0) | (xx <= 0)):
            raise ValueError("parameter inversion requires positive x and cum/ccum")
        result = _invert_parameter(p, q, xx, degrees, nc, solve_df=which == 3)
        if which == 3:
            degrees = _df(result, computed=True)
        else:
            nc = _noncentrality(result)
        _verify(p, q, xx, degrees, nc)
    return CDFNoncentralChiSquare(int(which), *map(_freeze, (p, q, xx, degrees, nc)))


def cum_nc_chisq(x: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Lower noncentral chi-square tail."""
    return cdf_nc_chisq(x=x, df=df, pnonc=pnonc).cum


def ccum_nc_chisq(x: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Direct upper noncentral chi-square tail."""
    return cdf_nc_chisq(x=x, df=df, pnonc=pnonc).ccum


def inv_nc_chisq(
    cum: ArrayLike | None, df: ArrayLike, pnonc: ArrayLike, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """Noncentral chi-square quantile preserving the smaller input tail."""
    return cdf_nc_chisq(2, cum=cum, ccum=ccum, df=df, pnonc=pnonc).x
