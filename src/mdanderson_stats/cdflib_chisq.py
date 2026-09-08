"""CDFLIB90 chi-square interfaces via its gamma distribution identity."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_gamma import _bounded, _invert_shape, _tails, cdf_gamma


def _df(value: ArrayLike | None, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError("df is required")
    result = finite(value, "df")
    if computed:
        tolerance = 8 * np.finfo(float).eps
        result = np.where((result < 1e-3) & (1e-3 - result <= tolerance * 1e-3), 1e-3, result)
        result = np.where((result > 1e10) & (result - 1e10 <= tolerance * 1e10), 1e10, result)
    if np.any((result < 1e-3) | (result > 1e10)):
        raise ValueError("df must lie in [1e-3,1e10]")
    return result


@dataclass(frozen=True)
class CDFChiSquare:
    """Computed group and immutable broadcast probability, x and df arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    x: FloatArray
    df: FloatArray


def cdf_chisq(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    x: ArrayLike | None = None,
    df: ArrayLike | None = None,
) -> CDFChiSquare:
    """Compute 1=tails, 2=x or 3=df; omit the computed group.

    Real degrees of freedom lie in [1e-3,1e10], x in [0,1e100].
    Probability pairs, numerical limits and errors follow cdf_gamma.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    if which == 3 and df is not None:
        raise ValueError("omit the parameter group being computed")
    if which == 3:
        xx = _bounded(x, "x")
        p, q = _pair(cum, ccum, "cum/ccum")
        p, q, xx = np.broadcast_arrays(p, q, xx)
        if np.any((p <= 0) | (q <= 0) | (xx <= 0)):
            raise ValueError("df inversion requires positive x and cum/ccum")
        # Preserve endpoint roots despite small inter-kernel probability rounding.
        lower, target = p <= q, np.minimum(p, q)
        adjusted = target.copy()
        for boundary in (0.0005, 5e9):
            lp, uq = _tails(np.full(p.shape, boundary), xx / 2)
            tail = np.where(lower, lp, uq)
            close = np.abs(target - tail) <= 32 * np.finfo(float).eps * tail
            adjusted = np.where(close, tail, adjusted)
        pp, qq = np.where(lower, adjusted, 1 - adjusted), np.where(lower, 1 - adjusted, adjusted)
        degrees = 2 * _invert_shape(pp, qq, xx / 2, 0.0005, 5e9)
        return CDFChiSquare(int(which), *map(_freeze, (p, q, xx, _df(degrees, computed=True))))
    result = cdf_gamma(which, cum=cum, ccum=ccum, x=x, shape=_df(df) / 2, rate=0.5)
    return CDFChiSquare(int(which), result.cum, result.ccum, result.x, _freeze(2 * result.shape))


def cum_chisq(x: ArrayLike, df: ArrayLike) -> FloatArray:
    """Chi-square lower tail with real degrees of freedom."""
    return cdf_chisq(x=x, df=df).cum


def ccum_chisq(x: ArrayLike, df: ArrayLike) -> FloatArray:
    """Direct chi-square upper tail."""
    return cdf_chisq(x=x, df=df).ccum


def inv_chisq(cum: ArrayLike | None, df: ArrayLike, *, ccum: ArrayLike | None = None) -> FloatArray:
    """Chi-square quantile; pass cum=None to supply only ccum."""
    return cdf_chisq(2, cum=cum, ccum=ccum, df=df).x
