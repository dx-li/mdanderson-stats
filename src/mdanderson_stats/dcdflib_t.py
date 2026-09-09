"""Legacy DCDFLIB Student's t with wide inputs and bounded df inversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaln, gammaln

from ._cdflib import _freeze
from ._dcdflib import _invert_df, _probability_pair
from ._validation import FloatArray, finite
from .cdflib_beta import _quantiles, _tails
from .dcdflib_f import _positive
from .dcdflib_normal import cdfnor


def _small_tail(t: FloatArray, df: FloatArray) -> FloatArray:
    result = np.full(t.shape, 0.5)
    # At the minimum subnormal df, the deviation from 1/2 is below one
    # probability spacing for every finite t; halving df itself rounds to zero.
    active = (t != 0) & (df / 2 > 0)
    tt, degrees = np.abs(t[active]), df[active]
    with np.errstate(over="ignore", invalid="ignore"):
        square = tt * tt
        total = degrees + square
        x, y = degrees / total, square / total
    logarithmic = ~np.isfinite(total) | (square == 0) | (x < np.finfo(float).tiny)
    logx = np.zeros(tt.shape)
    if np.any(logarithmic):
        ratio = 2 * np.log(tt[logarithmic]) - np.log(degrees[logarithmic])
        small = np.exp(-np.abs(ratio))
        x[logarithmic] = np.where(ratio >= 0, small / (1 + small), 1 / (1 + small))
        y[logarithmic] = np.where(ratio >= 0, 1 / (1 + small), small / (1 + small))
        logx[logarithmic] = -np.logaddexp(0, ratio)
    underflow = x < np.finfo(float).tiny
    tail = np.empty(tt.shape)
    aa = degrees / 2
    regular = ~underflow
    tail[regular] = (
        _tails(x[regular], y[regular], aa[regular], np.full(np.count_nonzero(regular), 0.5))[0] / 2
    )
    # I_x(a,1/2) = x**a/(a*B(a,1/2)) * (1+O(x)). For subnormal x,
    # the omitted relative correction is below float precision, while using
    # the rounded coordinate would lose significant relative precision.
    shapes = aa[underflow]
    normalizer = np.empty(shapes.shape)
    tiny_shape = shapes < 1
    normalizer[tiny_shape] = (
        gammaln(shapes[tiny_shape] + 1) + gammaln(0.5) - gammaln(shapes[tiny_shape] + 0.5)
    )
    normalizer[~tiny_shape] = np.log(shapes[~tiny_shape]) + betaln(shapes[~tiny_shape], 0.5)
    with np.errstate(over="ignore"):
        logs = shapes * logx[underflow] - normalizer - np.log(2.0)
    tail[underflow] = np.exp(logs)
    if np.any(~np.isfinite(tail) | (tail < 0) | (tail > 0.5)):
        raise ArithmeticError("legacy t tail evaluation failed")
    result[active] = tail
    return result


@dataclass(frozen=True)
class DCDFLIBStudentT:
    """Computed group and owned immutable p/q, t and df arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    t: FloatArray
    df: FloatArray


def cdft(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    t: ArrayLike | None = None,
    df: ArrayLike | None = None,
) -> DCDFLIBStudentT:
    """Compute 1=p/q, 2=t or 3=df for the legacy C/F77 interface.

    Input t is finite and df positive finite. Computed t lies in ±1e100;
    computed df lies in [1e-100,1e10]. Omit the computed group.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    if any(v is not None for v in {1: (p, q), 2: (t,), 3: (df,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if which != 2 and t is None:
        raise ValueError("t is required")
    tt = finite(0.0 if t is None else t, "t")
    degrees = _positive(df, "df") if which != 3 else np.asarray(1.0)
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, tt, degrees = np.broadcast_arrays(pp, qq, tt, degrees)
    if which == 1:
        small = _small_tail(tt, degrees)
        pp, qq = np.where(tt < 0, small, 1 - small), np.where(tt < 0, 1 - small, small)
    else:
        if np.any((pp <= 0) | (qq <= 0)):
            raise ValueError("legacy t inversion requires positive p and q")
        target = np.minimum(pp, qq)
        if which == 2:
            tt = np.zeros(pp.shape)
            normal = degrees >= 1e20
            # For representable probabilities |z|<39; the first t-quantile
            # correction relative to z is (z*z+1)/(4*df), below 4e-18 here.
            tt[normal] = cdfnor(2, p=pp[normal], q=qq[normal]).x
            active = (pp != qq) & ~normal
            if np.any(degrees[active] / 2 == 0):
                raise ValueError("legacy t quantile is not resolved within ±1e100")
            probability = 2 * target[active]
            x, y = _quantiles(
                probability, 1 - probability, degrees[active] / 2, np.full(probability.shape, 0.5)
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                magnitude = np.sqrt(degrees[active]) * np.sqrt(y / x)
                logs = (np.log(degrees[active]) + np.log(y) - np.log(x)) / 2
                magnitude = np.where(~np.isfinite(magnitude), np.exp(logs), magnitude)
            tt[active] = np.where(pp[active] < qq[active], -magnitude, magnitude)
            near = (np.abs(tt) > 1e100) & (np.abs(tt) - 1e100 <= 8 * np.finfo(float).eps * 1e100)
            tt = np.where(near, np.copysign(1e100, tt), tt)
            if np.any(~np.isfinite(tt) | (np.abs(tt) > 1e100)):
                raise ValueError("legacy t quantile lies outside [-1e100,1e100]")
        else:
            if np.any((tt == 0) | (pp == qq) | ((tt > 0) != (pp > qq))):
                raise ValueError("df inversion requires nonzero t and matching nonmedian p/q")
            fixed = tt.ravel()

            def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                return _small_tail(fixed[indices], value)

            degrees = _invert_df(
                target,
                np.full(pp.shape, 1e-100),
                np.full(pp.shape, 1e10),
                evaluate,
                unbracketed_message="legacy t df solution lies outside [1e-100,1e10]",
            )
        if np.any(
            np.abs(_small_tail(tt, degrees) - target) > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy t inversion failed forward verification")
    return DCDFLIBStudentT(int(which), *map(_freeze, (pp, qq, tt, degrees)))


def cumt(t: ArrayLike, df: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired legacy Student's t tails."""
    result = cdft(t=t, df=df)
    return result.p, result.q
