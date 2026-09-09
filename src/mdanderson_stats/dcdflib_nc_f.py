"""Legacy noncentral F contracts, including both df inversions and ignored q."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ncf

from ._cdflib import _freeze
from ._dcdflib import _invert_positive
from ._validation import FloatArray, finite
from .cdflib_nc_f import _invert_noncentrality, _refine_quantiles, _tails, _verify
from .dcdflib_f import _coordinate, _positive, cdff


@dataclass(frozen=True)
class DCDFLIBNoncentralF:
    """Computed group and immutable arrays; returned inverse q is 1-p."""

    which: int
    p: FloatArray
    q: FloatArray
    f: FloatArray
    dfn: FloatArray
    dfd: FloatArray
    pnonc: FloatArray


def cdffnc(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    f: ArrayLike | None = None,
    dfn: ArrayLike | None = None,
    dfd: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
) -> DCDFLIBNoncentralF:
    """Compute 1=p/q, 2=f, 3=dfn, 4=dfd or 5=pnonc, as in legacy C/F77.

    Inverse input p is required and in [0,1-1e-16]; q is ignored, including
    its shape. Returned inverse q is 1-p, rather than the native unused
    placeholder. Input df are positive; f and pnonc are nonnegative, with
    no upper input bounds. Computed f, df and pnonc retain legacy bounds.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4, 5)
    ):
        raise ValueError("which must be 1, 2, 3, 4 or 5")
    if any(v is not None for v in {1: (p, q), 2: (f,), 3: (dfn,), 4: (dfd,), 5: (pnonc,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if df_bracket is not None and which not in (3, 4):
        raise ValueError("df_bracket is only valid for df inversion")
    ff = _coordinate(f) if which != 2 else np.asarray(0.0)
    nn = _positive(dfn, "dfn") if which != 3 else np.asarray(1.0)
    dd = _positive(dfd, "dfd") if which != 4 else np.asarray(1.0)
    if which == 5:
        nc = np.asarray(0.0)
    else:
        if pnonc is None:
            raise ValueError("pnonc is required")
        nc = finite(pnonc, "pnonc")
        if np.any(nc < 0):
            raise ValueError("pnonc must be nonnegative")
    if which == 1:
        pp = np.asarray(0.5)
    else:
        if p is None:
            raise ValueError("p is required; the legacy noncentral F interface ignores q")
        pp = finite(p, "p")
        if np.any((pp < 0) | (pp > 1 - 1e-16)):
            raise ValueError("inverse p must be in [0,1-1e-16]")
    bounds = (1e-100, 1e100) if df_bracket is None else df_bracket
    if len(bounds) != 2:
        raise ValueError("df_bracket requires two endpoints")
    low, high = finite(bounds[0], "df lower bound"), finite(bounds[1], "df upper bound")
    if np.any((low < 1e-100) | (high > 1e100) | (low >= high)):
        raise ValueError("df_bracket requires ordered endpoints in [1e-100,1e100]")
    pp, ff, nn, dd, nc, low, high = np.broadcast_arrays(pp, ff, nn, dd, nc, low, high)
    qq = 1 - pp
    probability_atol = np.spacing(pp) / 2
    if which == 1:
        pp, qq = _tails(ff, nn, dd, nc)
    elif which == 2:
        central = nc == 0
        ff = np.empty(pp.shape)
        ff[central] = cdff(2, p=pp[central], dfn=nn[central], dfd=dd[central]).f
        lower, upper = (pp <= qq) & ~central, (pp > qq) & ~central
        try:
            ff[lower] = ncf.ppf(pp[lower], nn[lower], dd[lower], nc[lower])
        except OverflowError:
            ff[lower] = np.nan
        try:
            ff[upper] = ncf.isf(qq[upper], nn[upper], dd[upper], nc[upper])
        except OverflowError:
            ff[upper] = np.nan
        ff = _refine_quantiles(pp, qq, nn, dd, nc, np.where(pp == 0, 0.0, ff))
        ff = np.where((ff > 1e100) & (ff - 1e100 <= 8 * np.finfo(float).eps * 1e100), 1e100, ff)
        if np.any(~np.isfinite(ff) | (ff < 0) | (ff > 1e100)):
            raise ValueError("legacy noncentral F quantile lies outside [0,1e100]")
    else:
        if np.any((pp <= 0) | (ff <= 0)):
            raise ValueError("parameter inversion requires positive f and p")
        if which == 5:
            nc = _invert_noncentrality(pp, qq, ff, nn, dd, probability_atol=probability_atol)
        else:
            lower = (pp <= qq).ravel()
            x, n, d, noncentral = ff.ravel(), nn.ravel(), dd.ravel(), nc.ravel()

            def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(
                    x[indices],
                    value if which == 3 else n[indices],
                    d[indices] if which == 3 else value,
                    noncentral[indices],
                )
                return np.where(lower[indices], lp, uq)

            value = _invert_positive(
                np.minimum(pp, qq), low, high, evaluate, probability_atol=probability_atol
            )
            if which == 3:
                nn = value
            else:
                dd = value
    if which != 1:
        _verify(pp, qq, ff, nn, dd, nc, probability_atol=probability_atol)
    return DCDFLIBNoncentralF(int(which), *map(_freeze, (pp, qq, ff, nn, dd, nc)))


def cumfnc(
    f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike, pnonc: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Legacy paired noncentral F tails; pnonc is the squared-shift sum."""
    result = cdffnc(f=f, dfn=dfn, dfd=dfd, pnonc=pnonc)
    return result.p, result.q
