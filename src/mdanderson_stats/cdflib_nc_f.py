"""CDFLIB90 F95 noncentral F tails, quantiles and noncentrality inversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ncf

from ._cdflib import _freeze, _pair
from ._validation import FloatArray
from .cdflib_chisq import _df
from .cdflib_f import _coordinate, inv_f
from .cdflib_nc_chisq import _noncentrality
from .dcdflib_f import _tails as _central_tails


def _tails(
    f: FloatArray, dfn: FloatArray, dfd: FloatArray, nc: FloatArray
) -> tuple[FloatArray, FloatArray]:
    central = nc == 0
    p, q = np.empty(f.shape), np.empty(f.shape)
    p[central], q[central] = _central_tails(f[central], dfn[central], dfd[central])
    denominator_two = (dfd == 2) & ~central
    positive = denominator_two & (f > 0)
    p[denominator_two], q[denominator_two] = 0.0, 1.0
    # The Poisson-beta sum has this exact closed form when dfd=2.
    # Scale the first term by 1/f for small ratios, avoiding huge df
    # multiplied by an underflowed beta coordinate.
    logratio = np.log(2.0) - np.log(dfn[positive]) - np.log(f[positive])
    small = np.exp(-np.abs(logratio))
    correction = np.ones(small.shape)
    np.divide(np.log1p(small), small, out=correction, where=small > 0)
    first = np.empty(small.shape)
    left = logratio <= 0
    with np.errstate(over="ignore"):
        first[left] = correction[left] / f[positive][left]
        first[~left] = np.exp(
            np.log(dfn[positive][~left])
            - np.log(2.0)
            + np.log(logratio[~left] + np.log1p(small[~left]))
        )
        log_y = np.where(left, logratio, 0.0) - np.log1p(small)
        second = np.exp(np.log(nc[positive]) - np.log(2.0) + log_y)
        logp = -first - second
    p[positive], q[positive] = np.exp(logp), -np.expm1(logp)
    active = ~central & ~denominator_two
    p[active] = ncf.cdf(f[active], dfn[active], dfd[active], nc[active])
    q[active] = ncf.sf(f[active], dfn[active], dfd[active], nc[active])
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("noncentral F tail evaluation failed")
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


def _verify(
    p: FloatArray,
    q: FloatArray,
    f: FloatArray,
    dfn: FloatArray,
    dfd: FloatArray,
    nc: FloatArray,
    *,
    probability_atol: FloatArray | float = 0.0,
) -> None:
    lp, uq = _tails(f, dfn, dfd, nc)
    target = np.minimum(p, q)
    if np.any(
        np.abs(np.where(p <= q, lp, uq) - target)
        > 1e-7 * target + probability_atol + 32 * np.nextafter(0.0, 1.0)
    ):
        raise ArithmeticError("noncentral F inversion failed forward verification")


def _refine_quantiles(
    p: FloatArray,
    q: FloatArray,
    dfn: FloatArray,
    dfd: FloatArray,
    nc: FloatArray,
    guess: FloatArray,
) -> FloatArray:
    """Refine only inverse-kernel answers that fail the probability contract."""
    valid = np.isfinite(guess) & (guess >= 0) & (guess <= 1e100)
    lp, uq = _tails(np.where(valid, guess, 1.0), dfn, dfd, nc)
    target = np.minimum(p, q)
    failed = ~valid | (
        np.abs(np.where(p <= q, lp, uq) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
    )
    if not np.any(failed):
        return guess
    lower, target = (p <= q)[failed], target[failed]
    nn, dd, noncentral = dfn[failed], dfd[failed], nc[failed]

    def evaluate(f: FloatArray) -> FloatArray:
        lp, uq = _tails(f, nn, dd, noncentral)
        return np.where(lower, lp, uq)

    # Start with reciprocal bounds around f=1, the natural ratio scale;
    # widen below 1e-100 only when the requested tail requires it.
    low, high = np.full(target.shape, 1e-100), np.full(target.shape, 1e100)
    at_low, at_high = evaluate(low), evaluate(high)
    needs_smaller = np.where(lower, target < at_low, target > at_low)
    low = np.where(needs_smaller, np.nextafter(0.0, 1.0), low)
    at_low = evaluate(low)
    if np.any((target < np.minimum(at_low, at_high)) | (target > np.maximum(at_low, at_high))):
        raise ValueError("noncentral F quantile lies outside the representable f bounds")
    lo, hi = np.log(low), np.log(high)
    best, error = low.copy(), np.full(target.shape, np.inf)
    for _ in range(64):
        middle = (lo + hi) / 2
        candidate = np.exp(middle)
        value = evaluate(candidate)
        residual = np.abs(value - target)
        best = np.where(residual < error, candidate, best)
        error = np.minimum(error, residual)
        move_low = np.where(lower, value < target, value > target)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
    result = np.where(target == at_low, low, np.where(target == at_high, high, best))
    answer = guess.copy()
    answer[failed] = result
    return answer


def _invert_noncentrality(
    p: FloatArray,
    q: FloatArray,
    f: FloatArray,
    dfn: FloatArray,
    dfd: FloatArray,
    *,
    probability_atol: FloatArray | float = 0.0,
) -> FloatArray:
    lower, target = p <= q, np.minimum(p, q)

    def evaluate(nc: FloatArray) -> FloatArray:
        lp, uq = _tails(f, dfn, dfd, nc)
        return np.where(lower, lp, uq)

    lo, hi = np.zeros(p.shape), np.full(p.shape, 1e4)
    at_low, at_high = evaluate(lo), evaluate(hi)
    if np.any(at_low == at_high):
        raise ValueError("noncentrality is numerically unidentified at this f")
    adjusted = target.copy()
    for endpoint in (at_low, at_high):
        adjusted = np.where(
            np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint + probability_atol,
            endpoint,
            adjusted,
        )
    if np.any((adjusted < np.minimum(at_low, at_high)) | (adjusted > np.maximum(at_low, at_high))):
        raise ValueError("noncentrality solution lies outside [0,1e4]")
    for _ in range(64):
        middle = (lo + hi) / 2
        value = evaluate(middle)
        move_low = np.where(lower, value > adjusted, value < adjusted)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
    result = (lo + hi) / 2
    return np.where(adjusted == at_low, 0.0, np.where(adjusted == at_high, 1e4, result))


@dataclass(frozen=True)
class CDFNoncentralF:
    """Computed group and six owned immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    f: FloatArray
    dfn: FloatArray
    dfd: FloatArray
    pnonc: FloatArray


def cdf_nc_f(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    f: ArrayLike | None = None,
    dfn: ArrayLike | None = None,
    dfd: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
) -> CDFNoncentralF:
    """Compute 1=tails, 2=f or 3=pnonc, matching the F95 executable code.

    Omit the computed group. f is in [0,1e100], both df in [1e-3,1e10],
    pnonc in [0,1e4]. Legacy C/F77 df inversion modes remain separate scope.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    outputs = {1: (cum, ccum), 2: (f,), 3: (pnonc,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    ff = _coordinate(f) if which != 2 else np.asarray(0.0)
    nn, dd = _df(dfn), _df(dfd)
    nc = _noncentrality(pnonc) if which != 3 else np.asarray(0.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, ff, nn, dd, nc = np.broadcast_arrays(p, q, ff, nn, dd, nc)
    if which == 1:
        p, q = _tails(ff, nn, dd, nc)
    elif which == 2:
        if np.any(q <= 0):
            raise ValueError("zero ccum has no finite noncentral F quantile")
        central = nc == 0
        ff = np.empty(p.shape)
        ff[central] = inv_f(p[central], nn[central], dd[central], ccum=q[central])
        lower, upper = (p <= q) & ~central, (p > q) & ~central
        try:
            ff[lower] = ncf.ppf(p[lower], nn[lower], dd[lower], nc[lower])
        except OverflowError:
            ff[lower] = np.nan
        try:
            ff[upper] = ncf.isf(q[upper], nn[upper], dd[upper], nc[upper])
        except OverflowError:
            ff[upper] = np.nan
        ff = np.where(p == 0, 0.0, ff)
        ff = _coordinate(_refine_quantiles(p, q, nn, dd, nc, ff), computed=True)
        _verify(p, q, ff, nn, dd, nc)
    else:
        if np.any((ff <= 0) | (p <= 0) | (q <= 0)):
            raise ValueError("noncentrality inversion requires positive f and cum/ccum")
        nc = _noncentrality(_invert_noncentrality(p, q, ff, nn, dd))
        _verify(p, q, ff, nn, dd, nc)
    return CDFNoncentralF(int(which), *map(_freeze, (p, q, ff, nn, dd, nc)))


def cum_nc_f(f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Lower noncentral F tail."""
    return cdf_nc_f(f=f, dfn=dfn, dfd=dfd, pnonc=pnonc).cum


def ccum_nc_f(f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Direct upper noncentral F tail."""
    return cdf_nc_f(f=f, dfn=dfn, dfd=dfd, pnonc=pnonc).ccum


def inv_nc_f(
    cum: ArrayLike | None,
    dfn: ArrayLike,
    dfd: ArrayLike,
    pnonc: ArrayLike,
    *,
    ccum: ArrayLike | None = None,
) -> FloatArray:
    """Noncentral F quantile preserving the smaller probability tail."""
    return cdf_nc_f(2, cum=cum, ccum=ccum, dfn=dfn, dfd=dfd, pnonc=pnonc).f
