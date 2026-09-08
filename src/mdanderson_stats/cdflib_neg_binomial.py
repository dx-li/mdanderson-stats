"""CDFLIB90 negative-binomial interfaces with continuous failure/success counts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta import _invert_shape, _quantiles, _tails


def _count(value: ArrayLike | None, name: str) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    if np.any((result < 0) | (result > 1e10)):
        raise ValueError(f"{name} must lie in [0,1e10]")
    return result


@dataclass(frozen=True)
class CDFNegativeBinomial:
    """Computed group and six owned, immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    f: FloatArray
    s: FloatArray
    pr: FloatArray
    cpr: FloatArray


def cdf_neg_binomial(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    f: ArrayLike | None = None,
    s: ArrayLike | None = None,
    pr: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> CDFNegativeBinomial:
    """Compute 1=tails, 2=failures f, 3=successes s, or 4=pr/cpr.

    Omit the computed group. Counts are real in [0,1e10]. Zero required
    successes give cumulative probability one, including at success probability
    zero. Count inversions require interior success chance; a unit lower tail
    identifies zero required successes. Other count targets require positive tails;
    success-probability inversion requires a positive success count.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (f,), 3: (s,), 4: (pr, cpr)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    failures = _count(f, "f") if which != 2 else np.asarray(0.0)
    successes = _count(s, "s") if which != 3 else np.asarray(1.0)
    x, y = _pair(pr, cpr, "pr/cpr") if which != 4 else (np.asarray(0.5), np.asarray(0.5))
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, failures, successes, x, y = np.broadcast_arrays(p, q, failures, successes, x, y)
    if which == 1:
        p, q = _tails(x, y, np.where(successes == 0, 1, successes), failures + 1)
        p, q = np.where(successes == 0, 1, p), np.where(successes == 0, 0, q)
    elif which == 4:
        if np.any(successes == 0):
            raise ValueError("zero successes do not identify a unique success probability")
        x, y = _quantiles(p, q, successes, failures + 1)
    else:
        if np.any((p <= 0) | ((q <= 0) & (which == 2)) | (x <= 0) | (y <= 0)):
            raise ValueError("count inversion requires interior cum/ccum and pr/cpr")
        if which == 2 and np.any(successes == 0):
            raise ValueError("zero successes do not identify a failure count")
        solve_a = which == 3
        lo, hi = (np.nextafter(0.0, 1.0), 1e10) if solve_a else (1.0, 1e10 + 1)
        zero = (q == 0) & solve_a
        xx, yy = np.where(zero, 0.5, x), np.where(zero, 0.5, y)
        bb = np.where(zero, 1, failures + 1)
        lower, target = p <= q, np.where(zero, 0.5, np.minimum(p, q))
        adjusted = target.copy()
        # Match representable endpoint roots across a few units of tail rounding.
        for bound in (lo, hi):
            lp, uq = _tails(
                xx,
                yy,
                np.full(p.shape, bound) if solve_a else successes,
                bb if solve_a else np.full(p.shape, bound),
            )
            tail = np.where(lower, lp, uq)
            adjusted = np.where(
                np.abs(target - tail) <= 32 * np.finfo(float).eps * tail, tail, adjusted
            )
        pp, qq = np.where(lower, adjusted, 1 - adjusted), np.where(lower, 1 - adjusted, adjusted)
        shape = _invert_shape(pp, qq, xx, yy, successes, bb, solve_a=solve_a, lo=lo, hi=hi)
        if solve_a:
            successes = _count(np.where(zero, 0, shape), "s")
        else:
            failures = _count(shape - 1, "f")
    return CDFNegativeBinomial(int(which), *map(_freeze, (p, q, failures, successes, x, y)))


def cum_neg_binomial(
    f: ArrayLike, s: ArrayLike, pr: ArrayLike | None, *, cpr: ArrayLike | None = None
) -> FloatArray:
    """Lower tail for real failure and success counts."""
    return cdf_neg_binomial(f=f, s=s, pr=pr, cpr=cpr).cum


def ccum_neg_binomial(
    f: ArrayLike, s: ArrayLike, pr: ArrayLike | None, *, cpr: ArrayLike | None = None
) -> FloatArray:
    """Direct upper tail, retaining a separately supplied small failure chance."""
    return cdf_neg_binomial(f=f, s=s, pr=pr, cpr=cpr).ccum


def inv_neg_binomial(
    cum: ArrayLike | None,
    s: ArrayLike,
    pr: ArrayLike | None,
    *,
    ccum: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> FloatArray:
    """Continuous failure-count inverse; no integer rounding is performed."""
    return cdf_neg_binomial(2, cum=cum, ccum=ccum, s=s, pr=pr, cpr=cpr).f
