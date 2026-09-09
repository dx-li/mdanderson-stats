"""CDFLIB90 F95 F-distribution tails and quantiles."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta import _quantiles, _tails
from .cdflib_chisq import _df


def _coordinate(value: ArrayLike | None, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError("f is required")
    result = finite(value, "f")
    if computed:
        result = np.where(
            (result > 1e100) & (result - 1e100 <= 8 * np.finfo(float).eps * 1e100), 1e100, result
        )
    if np.any((result < 0) | (result > 1e100)):
        raise ValueError("f must lie in [0,1e100]")
    return result


@dataclass(frozen=True)
class CDFF:
    """Computed group and five owned immutable broadcast arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    f: FloatArray
    dfn: FloatArray
    dfd: FloatArray


def cdf_f(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    f: ArrayLike | None = None,
    dfn: ArrayLike | None = None,
    dfd: ArrayLike | None = None,
) -> CDFF:
    """Compute 1=tails or 2=f; omit the computed group, as in the F95 source.

    f lies in [0,1e100], numerator/denominator df in [1e-3,1e10].
    The older C/F77 degrees-of-freedom inversion modes are not this interface.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2)
    ):
        raise ValueError("which must be 1 or 2")
    if (which == 1 and (cum is not None or ccum is not None)) or (which == 2 and f is not None):
        raise ValueError("omit the parameter group being computed")
    nn, dd = _df(dfn), _df(dfd)
    ff = _coordinate(f) if which == 1 else np.asarray(0.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which == 2 else (np.asarray(0.5), np.asarray(0.5))
    p, q, ff, nn, dd = np.broadcast_arrays(p, q, ff, nn, dd)
    if which == 1:
        product = nn * ff
        total = product + dd
        p, q = _tails(product / total, dd / total, nn / 2, dd / 2)
    else:
        if np.any(q <= 0):
            raise ValueError("zero ccum has no finite F quantile")
        x, y = _quantiles(p, q, nn / 2, dd / 2)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ff = _coordinate((dd / nn) * (x / y), computed=True)
    return CDFF(int(which), *map(_freeze, (p, q, ff, nn, dd)))


def cum_f(f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> FloatArray:
    """Lower F tail with numerator and denominator degrees of freedom."""
    return cdf_f(f=f, dfn=dfn, dfd=dfd).cum


def ccum_f(f: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> FloatArray:
    """Direct upper F tail."""
    return cdf_f(f=f, dfn=dfn, dfd=dfd).ccum


def inv_f(
    cum: ArrayLike | None, dfn: ArrayLike, dfd: ArrayLike, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """F quantile preserving the smaller probability tail."""
    return cdf_f(2, cum=cum, ccum=ccum, dfn=dfn, dfd=dfd).f
