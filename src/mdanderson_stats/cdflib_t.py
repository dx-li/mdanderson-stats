"""CDFLIB90 Student's t tails, quantiles and bounded df inversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta import _quantiles, _tails
from .cdflib_chisq import _df


def _coordinate(value: ArrayLike | None, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError("t is required")
    result = finite(value, "t")
    if computed:
        near = (np.abs(result) > 1e100) & (
            np.abs(result) - 1e100 <= 8 * np.finfo(float).eps * 1e100
        )
        result = np.where(near, np.copysign(1e100, result), result)
    if np.any(np.abs(result) > 1e100):
        raise ValueError("t must lie in [-1e100,1e100]")
    return result


def _small_tail(t: FloatArray, df: FloatArray) -> FloatArray:
    square = t * t
    total = df + square
    p, _ = _tails(df / total, square / total, df / 2, np.full(df.shape, 0.5))
    return p / 2


@dataclass(frozen=True)
class CDFStudentT:
    """Computed group and four owned immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    t: FloatArray
    df: FloatArray


def cdf_t(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    t: ArrayLike | None = None,
    df: ArrayLike | None = None,
) -> CDFStudentT:
    """Compute 1=tails, 2=t or 3=df; omit the computed group.

    t lies in [-1e100,1e100], df in [1e-3,1e10]. Complementary probability
    inputs preserve the smaller tail. Median/indistinguishable df inversions
    and out-of-domain solutions fail explicitly.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    outputs = {1: (cum, ccum), 2: (t,), 3: (df,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    tt = _coordinate(t) if which != 2 else np.asarray(0.0)
    degrees = _df(df) if which != 3 else np.asarray(1.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, tt, degrees = np.broadcast_arrays(p, q, tt, degrees)
    if which == 1:
        small = _small_tail(tt, degrees)
        p, q = np.where(tt < 0, small, 1 - small), np.where(tt < 0, 1 - small, small)
    else:
        if np.any((p <= 0) | (q <= 0)):
            raise ValueError("t inversion requires positive cum and ccum")
        target = np.minimum(p, q)
        if which == 2:
            probability = 2 * target
            x, y = _quantiles(probability, 1 - probability, degrees / 2, np.full(p.shape, 0.5))
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                magnitude = np.sqrt(degrees) * np.sqrt(y / x)
            tt = _coordinate(np.where(p < q, -magnitude, magnitude), computed=True)
        else:
            if np.any((tt == 0) | (p == q) | ((tt > 0) != (p > q))):
                raise ValueError(
                    "df inversion requires nonzero t and a matching nonmedian probability"
                )
            low, high = np.full(p.shape, 1e-3), np.full(p.shape, 1e10)
            at_low, at_high = _small_tail(tt, low), _small_tail(tt, high)
            if np.any(at_low == at_high):
                raise ValueError("df is numerically unidentified at this t")
            adjusted = target.copy()
            for endpoint in (at_low, at_high):
                adjusted = np.where(
                    np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint,
                    endpoint,
                    adjusted,
                )
            if np.any((adjusted > at_low) | (adjusted < at_high)):
                raise ValueError("df solution lies outside [1e-3,1e10]")
            lo, hi = np.log(low), np.log(high)
            for _ in range(64):
                middle = (lo + hi) / 2
                move_low = _small_tail(tt, np.exp(middle)) > adjusted
                lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
            degrees = np.exp((lo + hi) / 2)
            degrees = np.where(
                adjusted == at_low, low, np.where(adjusted == at_high, high, degrees)
            )
            degrees = _df(degrees, computed=True)
            if np.any(
                np.abs(_small_tail(tt, degrees) - target)
                > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
            ):
                raise ArithmeticError("t df search failed forward verification")
    return CDFStudentT(int(which), *map(_freeze, (p, q, tt, degrees)))


def cum_t(t: ArrayLike, df: ArrayLike) -> FloatArray:
    """Lower Student's t tail."""
    return cdf_t(t=t, df=df).cum


def ccum_t(t: ArrayLike, df: ArrayLike) -> FloatArray:
    """Direct upper Student's t tail."""
    return cdf_t(t=t, df=df).ccum


def inv_t(cum: ArrayLike | None, df: ArrayLike, *, ccum: ArrayLike | None = None) -> FloatArray:
    """Student's t quantile using the smaller input probability."""
    return cdf_t(2, cum=cum, ccum=ccum, df=df).t
