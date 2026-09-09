"""CDFLIB90 binomial tails and inversions for real successes and trials."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray
from .cdflib_beta import _invert_shape, _quantiles, _tails
from .cdflib_neg_binomial import _count


def _forward(
    s: FloatArray, n: FloatArray, x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    at_end = s == n
    q, p = _tails(x, y, s + 1, np.where(at_end, 1, n - s))
    return np.where(at_end, 1, p), np.where(at_end, 0, q)


def _match_endpoint(target: FloatArray, endpoint: FloatArray) -> FloatArray:
    return np.where(
        np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint, endpoint, target
    )


@dataclass(frozen=True)
class CDFBinomial:
    """Computed group and six owned, immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    s: FloatArray
    n: FloatArray
    pr: FloatArray
    cpr: FloatArray


def cdf_binomial(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    s: ArrayLike | None = None,
    n: ArrayLike | None = None,
    pr: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> CDFBinomial:
    """Compute 1=tails, 2=success count, 3=trial count or 4=pr/cpr.

    Omit the computed group. Counts are real with 0 <= s <= n <= 1e10.
    This preserves the continuous beta extension, not an integer-valued PPF.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (s,), 3: (n,), 4: (pr, cpr)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    successes = _count(s, "s") if which != 2 else np.asarray(0.0)
    trials = _count(n, "n") if which != 3 else np.asarray(1.0)
    x, y = _pair(pr, cpr, "pr/cpr") if which != 4 else (np.asarray(0.5), np.asarray(0.5))
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, successes, trials, x, y = np.broadcast_arrays(p, q, successes, trials, x, y)
    if which in (1, 4) and np.any(successes > trials):
        raise ValueError("s must not exceed n")
    if which == 1:
        p, q = _forward(successes, trials, x, y)
    elif which == 4:
        if np.any(successes == trials):
            raise ValueError("s=n does not identify a unique success probability")
        x, y = _quantiles(q, p, successes + 1, trials - successes)
    else:
        endpoint = q == 0
        singleton = (trials == 0) if which == 2 else (successes == 1e10)
        if np.any(
            (p <= 0)
            | ((x == 0) & ~(singleton & endpoint))
            | ((y == 0) & ~endpoint)
            | (singleton & ~endpoint)
        ):
            raise ValueError("count request is unattainable or does not identify a unique count")
        lower, target = p <= q, np.minimum(p, q)
        if which == 2:
            p0, q0 = _forward(np.zeros(p.shape), trials, x, y)
            edge = np.where(lower, p0, q0)
            adjusted = _match_endpoint(target, edge)
            if np.any(np.where(lower, adjusted < edge, adjusted > edge)):
                raise ValueError("success solution lies outside [0,n]")
            lo, hi = np.zeros(p.shape), np.ones(p.shape)
            for _ in range(64):
                mid = (lo + hi) / 2
                lp, uq = _forward(trials * mid, trials, x, y)
                value = np.where(lower, lp, uq)
                move_low = np.where(lower, value < adjusted, value > adjusted)
                lo, hi = np.where(move_low, mid, lo), np.where(move_low, hi, mid)
            successes = trials * ((lo + hi) / 2)
            successes = np.where(adjusted == edge, 0, np.where(endpoint, trials, successes))
        else:
            pmax, qmax = _forward(successes, np.full(p.shape, 1e10), x, y)
            edge = np.where(lower, pmax, qmax)
            adjusted = _match_endpoint(target, edge)
            if np.any(np.where(lower, adjusted < edge, adjusted > edge)):
                raise ValueError("trial solution lies outside [s,1e10]")
            at_max = (adjusted == edge) & ~endpoint
            solved = endpoint | at_max
            xx, yy = np.where(solved, 0.5, x), np.where(solved, 0.5, y)
            aa = np.where(solved, 1, successes + 1)
            adjusted = np.where(solved, 0.5, adjusted)
            pp, qq = (
                np.where(lower, adjusted, 1 - adjusted),
                np.where(lower, 1 - adjusted, adjusted),
            )
            gap = _invert_shape(
                qq,
                pp,
                xx,
                yy,
                aa,
                np.ones(p.shape),
                solve_a=False,
                lo=np.nextafter(0.0, 1.0),
                hi=1e10,
            )
            trials = np.where(endpoint, successes, np.where(at_max, 1e10, successes + gap))
            trials = _count(trials, "n")
        lp, uq = _forward(successes, trials, x, y)
        matched = np.where(lower, lp, uq)
        if np.any(np.abs(matched - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)):
            raise ArithmeticError("binomial count search failed forward verification")
    return CDFBinomial(int(which), *map(_freeze, (p, q, successes, trials, x, y)))


def cum_binomial(
    s: ArrayLike, n: ArrayLike, pr: ArrayLike | None, *, cpr: ArrayLike | None = None
) -> FloatArray:
    """Inclusive binomial CDF extended to real success and trial counts."""
    return cdf_binomial(s=s, n=n, pr=pr, cpr=cpr).cum


def ccum_binomial(
    s: ArrayLike, n: ArrayLike, pr: ArrayLike | None, *, cpr: ArrayLike | None = None
) -> FloatArray:
    """Direct complementary binomial tail."""
    return cdf_binomial(s=s, n=n, pr=pr, cpr=cpr).ccum


def inv_binomial(
    cum: ArrayLike | None,
    n: ArrayLike,
    pr: ArrayLike | None,
    *,
    ccum: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> FloatArray:
    """Continuous success-count inverse without integer rounding."""
    return cdf_binomial(2, cum=cum, ccum=ccum, n=n, pr=pr, cpr=cpr).s
