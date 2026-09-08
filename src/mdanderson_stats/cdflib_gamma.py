"""CDFLIB90 gamma tails and inversions, with an explicit rate convention."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite


def _bounded(value: ArrayLike | None, name: str, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    lo = 0.0 if name == "x" else 1e-10
    if computed:
        tolerance = 8 * np.finfo(float).eps
        result = np.where((result < lo) & (lo - result <= tolerance * lo), lo, result)
        result = np.where((result > 1e100) & (result - 1e100 <= tolerance * 1e100), 1e100, result)
    if np.any((result < lo) | (result > 1e100)):
        raise ValueError(f"{name} must lie in [{lo:g},1e100]")
    return result


def _tails(shape: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = gammainc(shape, z), gammaincc(shape, z)
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("gamma tail evaluation failed")
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


@dataclass(frozen=True)
class CDFGamma:
    """Computed group and five independently owned, immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    x: FloatArray
    shape: FloatArray
    rate: FloatArray


def cdf_gamma(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    x: ArrayLike | None = None,
    shape: ArrayLike | None = None,
    rate: ArrayLike | None = None,
) -> CDFGamma:
    """Compute 1=tails, 2=x, 3=shape, or 4=rate; omit the computed group.

    The archived SCALE argument is a rate: the unit-rate coordinate is x*rate.
    Shape/rate lie in [1e-10,1e100], x in [0,1e100]. Input rate defaults to one.
    Inversion preserves the smaller probability tail. Shape/rate inversion
    requires positive x and both tails; upper-tail zero has no finite quantile.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (x,), 3: (shape,), 4: (rate,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    xx = _bounded(x, "x") if which != 2 else np.asarray(0.0)
    aa = _bounded(shape, "shape") if which != 3 else np.asarray(1.0)
    rr = _bounded(1.0 if rate is None else rate, "rate")
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, xx, aa, rr = np.broadcast_arrays(p, q, xx, aa, rr)
    if which == 1:
        p, q = _tails(aa, xx * rr)
    else:
        if np.any(q <= 0):
            raise ValueError("zero ccum has no finite gamma quantile")
        if which in (3, 4) and np.any((p <= 0) | (xx <= 0)):
            raise ValueError("shape/rate inversion requires positive x and cum/ccum")
        if which in (2, 4):
            lower = p <= q
            probability = np.minimum(p, q)
            z = np.where(lower, gammaincinv(aa, probability), gammainccinv(aa, probability))
            if np.any(~np.isfinite(z) | (z < 0)):
                raise ArithmeticError("gamma quantile evaluation failed")
            if which == 2:
                xx = _bounded(z / rr, "x", computed=True)
            else:
                with np.errstate(over="ignore"):
                    rr = _bounded(z / xx, "rate", computed=True)
        else:
            lower, target, z = p <= q, np.minimum(p, q), xx * rr

            def evaluate(a: FloatArray) -> FloatArray:
                lp, uq = _tails(a, z)
                return np.where(lower, lp, uq)

            low, high = np.full(p.shape, 1e-10), np.full(p.shape, 1e100)
            low_value, high_value = evaluate(low), evaluate(high)
            if np.any(
                (target < np.minimum(low_value, high_value))
                | (target > np.maximum(low_value, high_value))
            ):
                raise ValueError("gamma shape solution lies outside [1e-10,1e100]")
            log_low, log_high = np.log(low), np.log(high)
            for _ in range(64):
                middle = (log_low + log_high) / 2
                value = evaluate(np.exp(middle))
                move_low = np.where(lower, value > target, value < target)
                log_low = np.where(move_low, middle, log_low)
                log_high = np.where(move_low, log_high, middle)
            aa = np.exp((log_low + log_high) / 2)
            aa = np.where(target == low_value, low, np.where(target == high_value, high, aa))
            if np.any(np.abs(evaluate(aa) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)):
                raise ArithmeticError("gamma shape search failed forward verification")
    return CDFGamma(int(which), *map(_freeze, (p, q, xx, aa, rr)))


def cum_gamma(x: ArrayLike, shape: ArrayLike, rate: ArrayLike = 1) -> FloatArray:
    """Lower gamma tail with shape and rate (the reciprocal of scale)."""
    return cdf_gamma(x=x, shape=shape, rate=rate).cum


def ccum_gamma(x: ArrayLike, shape: ArrayLike, rate: ArrayLike = 1) -> FloatArray:
    """Direct upper gamma tail with shape and rate."""
    return cdf_gamma(x=x, shape=shape, rate=rate).ccum


def inv_gamma(
    cum: ArrayLike | None, shape: ArrayLike, rate: ArrayLike = 1, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """Gamma quantile; pass cum=None to supply only the complementary tail."""
    return cdf_gamma(2, cum=cum, ccum=ccum, shape=shape, rate=rate).x
