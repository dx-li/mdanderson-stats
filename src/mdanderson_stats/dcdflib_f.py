"""Legacy DCDFLIB F contracts, including both df inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._dcdflib import _invert_df as _search_df
from ._dcdflib import _probability_pair
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
    lower = (p <= q).ravel()
    ff, n, d = f.ravel(), nn.ravel(), dd.ravel()

    def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
        lp, uq = _tails(
            ff[indices], value if numerator else n[indices], d[indices] if numerator else value
        )
        return np.where(lower[indices], lp, uq)

    return _search_df(np.minimum(p, q), low, high, evaluate)


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
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
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
