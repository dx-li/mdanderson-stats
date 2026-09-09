"""Legacy DCDFLIB chi-square contracts over wide finite input domains."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import exp1

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray, finite
from .dcdflib_f import _positive
from .dcdflib_gamma import _exp1_log_inverse, _scaled_quantile
from .dcdflib_gamma import _tails as _gamma_tails


def _tails(x: FloatArray, df: FloatArray) -> tuple[FloatArray, FloatArray]:
    tiny_df = df < 2 * np.finfo(float).tiny
    p, q = np.zeros(x.shape), np.ones(x.shape)
    exponential = df == 2
    regular = ~tiny_df & ~exponential
    p[exponential] = -np.expm1(-x[exponential] / 2)
    q[exponential] = np.exp(-x[exponential] / 2)
    # P=x/2-x**2/8+... lies strictly below a half-ulp tie. If x/2
    # rounded upward in the subnormal regime, restore the lower neighbor.
    rounded_up = exponential & (x < 2 * np.finfo(float).tiny) & (2 * p > x)
    p[rounded_up] = np.nextafter(p[rounded_up], 0.0)
    p[regular], q[regular] = _gamma_tails(
        x[regular], df[regular] / 2, np.full(np.count_nonzero(regular), 0.5)
    )
    active = tiny_df & (x > 0)
    if np.any(active):
        xx = x[active]
        small = xx < 2 * np.finfo(float).tiny
        integral = np.empty(xx.shape)
        # E1(x/2) = -EulerGamma-log(x/2)+O(x); retain x before division.
        integral[small] = -0.5772156649015329 - np.log(xx[small]) + np.log(2.0)
        integral[~small] = exp1(xx[~small] / 2)
        # Multiplying df by E1/2 avoids underflow in the shape df/2.
        q[active] = df[active] * (integral / 2)
        p[active] = 1 - q[active]
    return p, q


def _quantile(p: FloatArray, q: FloatArray, df: FloatArray) -> FloatArray:
    tiny_df = df < 2 * np.finfo(float).tiny
    result = np.zeros(p.shape)
    regular = ~tiny_df
    result[regular] = _scaled_quantile(
        p[regular], q[regular], df[regular] / 2, np.full(np.count_nonzero(regular), 0.5)
    )
    active = tiny_df & (p > 0)
    if np.any(active):
        with np.errstate(over="ignore"):
            target = 2 * (q[active] / df[active])
        logz = -target - 0.5772156649015329
        normal = logz >= np.log(np.finfo(float).tiny)
        if np.any(normal):
            logz[normal] = _exp1_log_inverse(target[normal], logz[normal])
        result[active] = np.exp(logz + np.log(2.0))
    return result


@dataclass(frozen=True)
class DCDFLIBChiSquare:
    """Computed group and owned immutable p/q, x and df arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    x: FloatArray
    df: FloatArray


def cdfchi(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    x: ArrayLike | None = None,
    df: ArrayLike | None = None,
) -> DCDFLIBChiSquare:
    """Compute 1=p/q, 2=x or 3=df; omit the computed group.

    Inputs x>=0 and df>0 are finite with no upper cap. Computed x lies in
    [0,1e100], computed df in [1e-100,1e100]. Inverse q must be positive.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    if any(v is not None for v in {1: (p, q), 2: (x,), 3: (df,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if which != 2 and x is None:
        raise ValueError("x is required")
    xx = finite(0.0 if x is None else x, "x")
    if np.any(xx < 0):
        raise ValueError("x must be nonnegative")
    degrees = _positive(df, "df") if which != 3 else np.asarray(1.0)
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, xx, degrees = np.broadcast_arrays(pp, qq, xx, degrees)
    if which == 1:
        pp, qq = _tails(xx, degrees)
    else:
        if np.any(qq <= 0):
            raise ValueError("legacy chi-square inversion requires positive q")
        if which == 2:
            xx = _quantile(pp, qq, degrees)
            xx = np.where((xx > 1e100) & (xx <= 1e100 * (1 + 8 * np.finfo(float).eps)), 1e100, xx)
            if np.any(~np.isfinite(xx) | (xx < 0) | (xx > 1e100) | ((xx == 0) & (pp > 0))):
                raise ValueError("chi-square x is not representable within [0,1e100]")
        else:
            if np.any((pp <= 0) | (xx <= 0)):
                raise ValueError("df inversion requires positive x and p/q")
            fixed, lower = xx.ravel(), (pp <= qq).ravel()

            def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(fixed[indices], value)
                return np.where(lower[indices], lp, uq)

            degrees = _invert_positive(
                np.minimum(pp, qq),
                np.full(pp.shape, 1e-100),
                np.full(pp.shape, 1e100),
                evaluate,
                initial=np.where(xx >= 1, xx, 5.0),
                unbracketed_message="chi-square df solution lies outside [1e-100,1e100]",
            )
        lp, uq = _tails(xx, degrees)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy chi-square inversion failed forward verification")
    return DCDFLIBChiSquare(int(which), *map(_freeze, (pp, qq, xx, degrees)))


def cumchi(x: ArrayLike, df: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired chi-square tails for nonnegative finite x and positive finite df."""
    result = cdfchi(x=x, df=df)
    return result.p, result.q
