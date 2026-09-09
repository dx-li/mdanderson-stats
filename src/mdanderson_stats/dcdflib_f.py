"""Legacy DCDFLIB F contracts, including both df inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta import _quantiles
from .cdflib_beta import _tails as _beta_tails


def _positive(value: ArrayLike | None, name: str) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def _coordinate(value: ArrayLike | None) -> FloatArray:
    if value is None:
        raise ValueError("f is required")
    result = finite(value, "f")
    if np.any(result < 0):
        raise ValueError("f must be nonnegative")
    return result


def _tails(f: FloatArray, nn: FloatArray, dd: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = np.zeros(f.shape), np.ones(f.shape)
    symmetric = (f == 1) & (nn == dd)
    p[symmetric] = q[symmetric] = 0.5
    active = (f > 0) & ~symmetric
    ff, n, d = f[active], nn[active], dd[active]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = n * ff
        total = product + d
        x, y = product / total, d / total
    needs_logs = ~np.isfinite(total) | (product == 0)
    if np.any(needs_logs):
        ratio = np.log(ff[needs_logs]) + np.log(n[needs_logs]) - np.log(d[needs_logs])
        small = np.exp(-np.abs(ratio))
        small = small / (1 + small)
        x[needs_logs] = np.where(ratio <= 0, small, 1 - small)
        y[needs_logs] = np.where(ratio <= 0, 1 - small, small)
    if np.any((x <= 0) | (y <= 0) | (n / 2 == 0) | (d / 2 == 0)):
        raise ArithmeticError("legacy F beta coordinates or shapes underflowed")
    p[active], q[active] = _beta_tails(x, y, n / 2, d / 2)
    return p, q


def _verify(p: FloatArray, q: FloatArray, f: FloatArray, nn: FloatArray, dd: FloatArray) -> None:
    lp, uq = _tails(f, nn, dd)
    target = np.minimum(p, q)
    if np.any(
        np.abs(np.where(p <= q, lp, uq) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
    ):
        raise ArithmeticError("legacy F inversion failed forward verification")


def _invert_df(
    p: FloatArray,
    q: FloatArray,
    f: FloatArray,
    nn: FloatArray,
    dd: FloatArray,
    low: FloatArray,
    high: FloatArray,
    *,
    numerator: bool,
) -> FloatArray:
    lower, target = p <= q, np.minimum(p, q)

    def evaluate(value: FloatArray) -> FloatArray:
        lp, uq = _tails(f, value if numerator else nn, dd if numerator else value)
        return np.where(lower, lp, uq)

    at_low, at_high = evaluate(low), evaluate(high)
    if np.any(at_low == at_high):
        raise ValueError("df is numerically unidentified across this bracket")
    adjusted = target.copy()
    for endpoint in (at_low, at_high):
        adjusted = np.where(
            np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint, endpoint, adjusted
        )
    if np.any((adjusted < np.minimum(at_low, at_high)) | (adjusted > np.maximum(at_low, at_high))):
        raise ValueError("F df root is not bracketed; multiple roots may require df_bracket")
    resolved = (adjusted == at_low) | (adjusted == at_high)
    answer = np.where(adjusted == at_low, low, high)
    if np.all(resolved):
        return answer
    active = ~resolved
    f, nn, dd = f[active], nn[active], dd[active]
    lower, adjusted = lower[active], adjusted[active]
    low, high, at_low = low[active], high[active], at_low[active]
    # Establish a local crossing from the legacy initial value of five.
    # Huge log-midpoints can reach ill-conditioned beta parameters even
    # when the desired df is ordinary and nearby.
    current = np.clip(np.full(low.shape, 5.0), low, high)
    current_residual = evaluate(current) - adjusted
    left = np.signbit(current_residual) != np.signbit(at_low - adjusted)
    found = current_residual == 0
    bracket_low, bracket_high = current.copy(), current.copy()
    for _ in range(300):
        if np.all(found):
            break
        trial = np.where(left, np.maximum(low, current / 5), np.minimum(high, current * 5))
        trial = np.where(found, current, trial)
        residual = evaluate(trial) - adjusted
        crossed = (np.signbit(residual) != np.signbit(current_residual)) | (residual == 0)
        newly_found = ~found & crossed
        bracket_low = np.where(newly_found, np.minimum(current, trial), bracket_low)
        bracket_high = np.where(newly_found, np.maximum(current, trial), bracket_high)
        found |= crossed
        current, current_residual = trial, residual
    if not np.all(found):
        raise ArithmeticError("legacy F df search could not isolate a crossing")
    lo, hi = np.log(bracket_low), np.log(bracket_high)
    residual_low = evaluate(bracket_low) - adjusted
    best, error = bracket_low.copy(), np.abs(residual_low)
    for _ in range(64):
        middle = (lo + hi) / 2
        candidate = np.clip(np.exp(middle), bracket_low, bracket_high)
        residual = evaluate(candidate) - adjusted
        best = np.where(np.abs(residual) < error, candidate, best)
        error = np.minimum(error, np.abs(residual))
        move_low = np.signbit(residual) == np.signbit(residual_low)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
        residual_low = np.where(move_low, residual, residual_low)
    answer[active] = best
    return answer


@dataclass(frozen=True)
class DCDFLIBF:
    """Computed group and immutable legacy p/q, f and df arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    f: FloatArray
    dfn: FloatArray
    dfd: FloatArray


def cdff(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    f: ArrayLike | None = None,
    dfn: ArrayLike | None = None,
    dfd: ArrayLike | None = None,
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
) -> DCDFLIBF:
    """Compute 1=p/q, 2=f, 3=dfn or 4=dfd for the C/F77 DCDFLIB interface.

    Finite input df are positive and input f nonnegative. Computed f is
    bounded by 1e100; df searches retain [1e-100,1e100]. An explicit df_bracket
    selects a sign-changing interval when the CDF has multiple df roots.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (f,), 3: (dfn,), 4: (dfd,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if df_bracket is not None and which not in (3, 4):
        raise ValueError("df_bracket is only valid for df inversion")
    ff = _coordinate(f) if which != 2 else np.asarray(0.0)
    nn = _positive(dfn, "dfn") if which != 3 else np.asarray(1.0)
    dd = _positive(dfd, "dfd") if which != 4 else np.asarray(1.0)
    pp, qq = _pair(p, q, "p/q") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    if which != 1 and p is not None and q is not None:
        raw_p, raw_q = finite(p, "p"), finite(q, "q")
        if np.any(np.abs(raw_p + raw_q - 1) > 3 * np.finfo(float).eps):
            raise ValueError("p and q must sum to one within three machine epsilons")
    bounds = (1e-100, 1e100) if df_bracket is None else df_bracket
    if len(bounds) != 2:
        raise ValueError("df_bracket requires two endpoints")
    low, high = finite(bounds[0], "df lower bound"), finite(bounds[1], "df upper bound")
    if np.any((low < 1e-100) | (high > 1e100) | (low >= high)):
        raise ValueError("df_bracket requires ordered endpoints in [1e-100,1e100]")
    pp, qq, ff, nn, dd, low, high = np.broadcast_arrays(pp, qq, ff, nn, dd, low, high)
    if which == 1:
        pp, qq = _tails(ff, nn, dd)
    else:
        if np.any(qq <= 0):
            raise ValueError("legacy F inversion requires positive q")
        if which == 2:
            x, y = _quantiles(pp, qq, nn / 2, dd / 2)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
                ff = (dd / nn) * (x / y)
                logs = np.log(dd) - np.log(nn) + np.log(x) - np.log(y)
                ff = np.where(~np.isfinite(ff) | ((ff == 0) & (pp > 0)), np.exp(logs), ff)
            ff = np.where((ff > 1e100) & (ff - 1e100 <= 8 * np.finfo(float).eps * 1e100), 1e100, ff)
            if np.any(~np.isfinite(ff) | (ff < 0) | (ff > 1e100)):
                raise ValueError("legacy F quantile lies outside [0,1e100]")
        else:
            if np.any((pp <= 0) | (ff <= 0)):
                raise ValueError("F df inversion requires positive f and p/q")
            value = _invert_df(pp, qq, ff, nn, dd, low, high, numerator=which == 3)
            if which == 3:
                nn = value
            else:
                dd = value
        _verify(pp, qq, ff, nn, dd)
    return DCDFLIBF(int(which), *map(_freeze, (pp, qq, ff, nn, dd)))


def cumf(f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Legacy paired lower/upper F tails."""
    result = cdff(f=f, dfn=dfn, dfd=dfd)
    return result.p, result.q
