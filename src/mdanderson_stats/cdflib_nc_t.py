"""CDFLIB90 noncentral t tails and bracketed parameter inversions."""

from dataclasses import dataclass
from math import exp, log, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import gammainc
from scipy.stats import nct

from ._cdflib import _freeze, _pair
from ._validation import FloatArray
from .cdflib_chisq import _df
from .cdflib_nc_chisq import _noncentrality
from .cdflib_normal import cdf_normal
from .cdflib_t import _coordinate, cdf_t, inv_t


def _negative_tail(t: float, df: float, nc: float) -> float:
    """Condition on the normal numerator to avoid subtracting small tails."""
    # P(T<=t) <= Phi(-nc) for t<0, so a normal-tail underflow is decisive.
    if float(cdf_normal(x=-nc).cum) == 0:
        return 0.0
    scale = nc + 1
    prefactor = exp(-nc * nc / 2 - 0.5 * log(2 * pi)) / scale

    def integrand(s: float) -> float:
        u = s / scale
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            ratio = np.float64(u) * sqrt(df / 2) / abs(t)
            probability = gammainc(df / 2, ratio * ratio)
        return exp(-nc * u - u * u / 2) * float(probability)

    result = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=2e-11, limit=200, full_output=1)
    value, error = result[:2]
    if (
        len(result) != 3
        or not np.isfinite(value)
        or not np.isfinite(error)
        or error > 1e-8 * abs(value)
    ):
        raise ArithmeticError("noncentral t negative-tail quadrature failed")
    return prefactor * value


def _tails(t: FloatArray, df: FloatArray, nc: FloatArray) -> tuple[FloatArray, FloatArray]:
    central = nc == 0
    zero = (t == 0) & ~central
    active = ~(central | zero)
    p, q = np.empty(t.shape), np.empty(t.shape)
    r = cdf_t(t=t[central], df=df[central])
    p[central], q[central] = r.cum, r.ccum
    normal = cdf_normal(x=-nc[zero])
    p[zero], q[zero] = normal.cum, normal.ccum
    p[active] = nct.cdf(t[active], df[active], nc[active])
    q[active] = nct.sf(t[active], df[active], nc[active])
    fallback = (t < 0) & (nc > 0) & ((p < 1e-6) | ~np.isfinite(p))
    for index in np.flatnonzero(fallback):
        p.flat[index] = _negative_tail(
            float(t.flat[index]), float(df.flat[index]), float(nc.flat[index])
        )
        q.flat[index] = 1 - p.flat[index]
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("noncentral t tail evaluation failed")
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


def _verify(p: FloatArray, q: FloatArray, t: FloatArray, df: FloatArray, nc: FloatArray) -> None:
    lp, uq = _tails(t, df, nc)
    target = np.minimum(p, q)
    if np.any(
        np.abs(np.where(p <= q, lp, uq) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
    ):
        raise ArithmeticError("noncentral t inversion failed forward verification")


def _invert_parameter(
    p: FloatArray,
    q: FloatArray,
    t: FloatArray,
    df: FloatArray,
    nc: FloatArray,
    low: FloatArray,
    high: FloatArray,
    *,
    solve_df: bool,
) -> FloatArray:
    lower, target = p <= q, np.minimum(p, q)

    def evaluate(value: FloatArray) -> FloatArray:
        lp, uq = _tails(t, value if solve_df else df, nc if solve_df else value)
        return np.where(lower, lp, uq)

    at_low, at_high = evaluate(low), evaluate(high)
    if np.any(at_low == at_high):
        raise ValueError("parameter is numerically unidentified across this bracket")
    adjusted = target.copy()
    for endpoint in (at_low, at_high):
        adjusted = np.where(
            np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint, endpoint, adjusted
        )
    if np.any((adjusted < np.minimum(at_low, at_high)) | (adjusted > np.maximum(at_low, at_high))):
        if solve_df:
            raise ValueError(
                "df root is not bracketed; df may have multiple roots: supply df_bracket"
            )
        raise ValueError("noncentrality solution lies outside [0,1e4]")
    resolved = (adjusted == at_low) | (adjusted == at_high)
    answer = np.where(adjusted == at_low, low, high)
    if np.all(resolved):
        return answer
    active = ~resolved
    t, df, nc = t[active], df[active], nc[active]
    lower, adjusted = lower[active], adjusted[active]
    low, high, at_low = low[active], high[active], at_low[active]
    lo, hi = (np.log(low), np.log(high)) if solve_df else (low.copy(), high.copy())
    low_residual = at_low - adjusted
    best, error = low.copy(), np.abs(low_residual)
    # Preserve a sign-changing bracket, without assuming df monotonicity.
    for _ in range(64):
        middle = (lo + hi) / 2
        candidate = np.exp(middle) if solve_df else middle
        residual = evaluate(candidate) - adjusted
        best = np.where(np.abs(residual) < error, candidate, best)
        error = np.minimum(error, np.abs(residual))
        move_low = np.signbit(residual) == np.signbit(low_residual)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
        low_residual = np.where(move_low, residual, low_residual)
    answer[active] = best
    return answer


@dataclass(frozen=True)
class CDFNoncentralT:
    """Computed group and five owned immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    t: FloatArray
    df: FloatArray
    pnonc: FloatArray


def cdf_nc_t(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    t: ArrayLike | None = None,
    df: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
) -> CDFNoncentralT:
    """Compute 1=tails, 2=t, 3=df or 4=pnonc; omit the computed group.

    t is in [-1e100,1e100], df in [1e-3,1e10], pnonc in [0,1e4].
    df_bracket optionally selects a df root using a sign-changing interval;
    the default is the full df domain, as in the source. Roots are not enumerated.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (t,), 3: (df,), 4: (pnonc,)}
    if any(v is not None for v in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    if df_bracket is not None and which != 3:
        raise ValueError("df_bracket is only valid for df inversion")
    tt = _coordinate(t) if which != 2 else np.asarray(0.0)
    degrees = _df(df) if which != 3 else np.asarray(1.0)
    nc = _noncentrality(pnonc) if which != 4 else np.asarray(0.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    if which == 3:
        bounds = (0.001, 1e10) if df_bracket is None else df_bracket
        if len(bounds) != 2:
            raise ValueError("df_bracket requires two endpoints")
        low, high = _df(bounds[0]), _df(bounds[1])
        if np.any(low >= high):
            raise ValueError("df_bracket lower endpoint must be below its upper endpoint")
    else:
        low, high = np.asarray(0.0), np.asarray(1e4)
    p, q, tt, degrees, nc, low, high = np.broadcast_arrays(p, q, tt, degrees, nc, low, high)
    if which == 1:
        p, q = _tails(tt, degrees, nc)
    else:
        if np.any((p <= 0) | (q <= 0)):
            raise ValueError("noncentral t inversion requires positive cum and ccum")
        if which == 2:
            central = nc == 0
            tt = np.empty(p.shape)
            tt[central] = inv_t(p[central], degrees[central], ccum=q[central])
            lower, upper = (p <= q) & ~central, (p > q) & ~central
            tt[lower] = nct.ppf(p[lower], degrees[lower], nc[lower])
            tt[upper] = nct.isf(q[upper], degrees[upper], nc[upper])
            tt = _coordinate(tt, computed=True)
        else:
            if which == 3 and np.any(tt == 0):
                raise ValueError("df is unidentified at t=0")
            result = _invert_parameter(p, q, tt, degrees, nc, low, high, solve_df=which == 3)
            if which == 3:
                degrees = _df(result, computed=True)
            else:
                nc = _noncentrality(result)
        _verify(p, q, tt, degrees, nc)
    return CDFNoncentralT(int(which), *map(_freeze, (p, q, tt, degrees, nc)))


def cum_nc_t(t: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Lower noncentral t tail; pnonc is a normal mean, not its square."""
    return cdf_nc_t(t=t, df=df, pnonc=pnonc).cum


def ccum_nc_t(t: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> FloatArray:
    """Direct upper noncentral t tail."""
    return cdf_nc_t(t=t, df=df, pnonc=pnonc).ccum


def inv_nc_t(
    cum: ArrayLike | None, df: ArrayLike, pnonc: ArrayLike, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """Noncentral t quantile preserving the smaller input tail."""
    return cdf_nc_t(2, cum=cum, ccum=ccum, df=df, pnonc=pnonc).t
