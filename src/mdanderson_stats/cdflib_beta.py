"""CDFLIB90 beta tails, quantiles and bounded shape inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincc, betainccinv, betaincinv

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite


def _shape(value: ArrayLike | None, name: str) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    if np.any((result < 1e-10) | (result > 1e10)):
        raise ValueError(f"{name} must lie in [1e-10,1e10]")
    return result


def _tails(
    x: FloatArray, cx: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray]:
    left = x <= cx
    aa, bb, z = np.where(left, a, b), np.where(left, b, a), np.minimum(x, cx)
    lower, upper = betainc(aa, bb, z), betaincc(aa, bb, z)
    p, q = np.where(left, lower, upper), np.where(left, upper, lower)
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("beta tail evaluation failed")
    symmetric_midpoint = (a == b) & (x == cx)
    p = np.where(symmetric_midpoint, 0.5, p)
    q = np.where(symmetric_midpoint, 0.5, q)
    # Independent kernel rounding must not produce an inconsistent output pair.
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


@dataclass(frozen=True)
class CDFBeta:
    """All six broadcast arrays; which identifies the computed parameter group."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    x: FloatArray
    cx: FloatArray
    a: FloatArray
    b: FloatArray


def cdf_beta(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    x: ArrayLike | None = None,
    cx: ArrayLike | None = None,
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
) -> CDFBeta:
    """Solve beta CDFLIB90 groups: 1=tails, 2=x/cx, 3=a, 4=b.

    Supply all input groups and omit the computed group. Either or both members
    of a complementary pair are accepted. Shapes and shape searches use the
    original inclusive [1e-10,1e10] domain. Degenerate/unidentifiable shape
    inversions and out-of-range solutions raise ValueError, never clipped roots.
    Every result array broadcasts, is independently owned and is read-only.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (x, cx), 3: (a,), 4: (b,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    aa = _shape(a, "a") if which != 3 else np.asarray(1.0)
    bb = _shape(b, "b") if which != 4 else np.asarray(1.0)
    xx, yy = _pair(x, cx, "x/cx") if which != 2 else (np.asarray(0.5), np.asarray(0.5))
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, xx, yy, aa, bb = np.broadcast_arrays(p, q, xx, yy, aa, bb)
    if which == 1:
        p, q = _tails(xx, yy, aa, bb)
    elif which == 2:
        # Invert the smaller probability and compute both coordinates directly.
        left = p <= q
        first, second, probability = (
            np.where(left, aa, bb),
            np.where(left, bb, aa),
            np.minimum(p, q),
        )
        z = betaincinv(first, second, probability)
        cz = betainccinv(second, first, probability)
        xx, yy = np.where(left, z, cz), np.where(left, cz, z)
        if np.any(~np.isfinite(xx) | ~np.isfinite(yy)):
            raise ArithmeticError("beta quantile evaluation failed")
    else:
        if np.any((xx <= 0) | (yy <= 0) | (p <= 0) | (q <= 0)):
            raise ValueError("shape inversion requires interior x/cx and cum/ccum")
        lower_tail = p <= q
        target = np.minimum(p, q)
        increasing = ~lower_tail if which == 3 else lower_tail

        def evaluate(shape: FloatArray) -> FloatArray:
            lp, uq = _tails(xx, yy, shape if which == 3 else aa, shape if which == 4 else bb)
            return np.where(lower_tail, lp, uq)

        low, high = np.full(p.shape, 1e-10), np.full(p.shape, 1e10)
        low_value, high_value = evaluate(low), evaluate(high)
        if np.any(
            (target < np.minimum(low_value, high_value))
            | (target > np.maximum(low_value, high_value))
        ):
            raise ValueError("beta shape solution lies outside [1e-10,1e10]")
        log_low, log_high = np.log(low), np.log(high)
        # A monotone batched search in log shape retains extreme complements.
        for _ in range(64):
            middle = (log_low + log_high) / 2
            value = evaluate(np.exp(middle))
            move_low = np.where(increasing, value < target, value > target)
            log_low = np.where(move_low, middle, log_low)
            log_high = np.where(move_low, log_high, middle)
        shape = np.exp((log_low + log_high) / 2)
        shape = np.where(target == low_value, low, np.where(target == high_value, high, shape))
        matched = evaluate(shape)
        if np.any(np.abs(matched - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)):
            raise ArithmeticError("beta shape search failed forward verification")
        if which == 3:
            aa = shape
        else:
            bb = shape
    return CDFBeta(int(which), *map(_freeze, (p, q, xx, yy, aa, bb)))


def cum_beta(
    x: ArrayLike | None, a: ArrayLike, b: ArrayLike, *, cx: ArrayLike | None = None
) -> FloatArray:
    """Lower beta tail; pass x=None to provide only its complement cx."""
    return cdf_beta(x=x, cx=cx, a=a, b=b).cum


def ccum_beta(
    x: ArrayLike | None, a: ArrayLike, b: ArrayLike, *, cx: ArrayLike | None = None
) -> FloatArray:
    """Upper beta tail calculated directly, avoiding subtraction from one."""
    return cdf_beta(x=x, cx=cx, a=a, b=b).ccum


def inv_beta(
    cum: ArrayLike | None, a: ArrayLike, b: ArrayLike, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """Beta quantile; cdf_beta(which=2) additionally returns its complement."""
    return cdf_beta(2, cum=cum, ccum=ccum, a=a, b=b).x
