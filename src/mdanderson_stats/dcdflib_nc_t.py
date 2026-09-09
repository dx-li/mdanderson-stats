"""Legacy signed noncentral t with checked, bounded parameter inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import nct

from ._cdflib import _freeze
from ._dcdflib import _invert_positive
from ._dcdflib_nc_t import _tails
from ._validation import FloatArray, finite
from .dcdflib_f import _positive
from .dcdflib_normal import cdfnor
from .dcdflib_t import cdft


@dataclass(frozen=True)
class DCDFLIBNoncentralT:
    """Computed group and owned immutable arrays; inverse q is 1-p."""

    which: int
    p: FloatArray
    q: FloatArray
    t: FloatArray
    df: FloatArray
    pnonc: FloatArray


def cdftnc(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    t: ArrayLike | None = None,
    df: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
) -> DCDFLIBNoncentralT:
    """Compute 1=p/q, 2=t, 3=df or 4=pnonc; omit the computed group.

    Input t/pnonc are signed finite and df positive finite. Computed t is
    in [-1e100,1e100], df [1e-100,1e4], and pnonc [-1e4,1e4]. Inversions
    require p in [0,1-1e-16]; q is ignored including its type and shape.
    df_bracket selects a crossing within the executable's df bounds.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (t,), 3: (df,), 4: (pnonc,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if df_bracket is not None and which != 3:
        raise ValueError("df_bracket is only valid for df inversion")
    if which != 2 and t is None:
        raise ValueError("t is required")
    if which != 4 and pnonc is None:
        raise ValueError("pnonc is required")
    tt = finite(0.0 if t is None else t, "t")
    dd = _positive(df, "df") if which != 3 else np.asarray(1.0)
    nn = finite(0.0 if pnonc is None else pnonc, "pnonc")
    if which == 1:
        pp = np.asarray(0.5)
    else:
        if p is None:
            raise ValueError("p is required; the legacy interface ignores q")
        pp = finite(p, "p")
        if np.any((pp < 0) | (pp > 1 - 1e-16)):
            raise ValueError("inverse p must be in [0,1-1e-16]")
    bounds = (1e-100, 1e4) if df_bracket is None else df_bracket
    if len(bounds) != 2:
        raise ValueError("df_bracket requires two endpoints")
    low, high = finite(bounds[0], "lower df"), finite(bounds[1], "upper df")
    if np.any((low < 1e-100) | (high > 1e4) | (low >= high)):
        raise ValueError("df_bracket must increase within [1e-100,1e4]")
    pp, tt, dd, nn, low, high = np.broadcast_arrays(pp, tt, dd, nn, low, high)
    qq = 1 - pp
    tolerance = np.spacing(pp) / 2
    if which == 1:
        pp, qq = _tails(tt, dd, nn)
    else:
        if np.any(pp == 0):
            raise ValueError("no finite noncentral t inverse is identified by p=0")
        left = (pp <= qq).ravel()
        if which == 3:
            if np.any(tt == 0):
                raise ValueError("df is unidentified at t=0")
            fixed_t, fixed_nc = tt.ravel(), nn.ravel()

            def evaluate_df(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(fixed_t[indices], v, fixed_nc[indices])
                return np.where(left[indices], lp, uq)

            dd = _invert_positive(
                np.minimum(pp, qq), low, high, evaluate_df, probability_atol=tolerance
            )
        else:
            value = np.zeros(pp.shape)
            direct = (nn == 0) if which == 2 else (tt == 0)
            if np.any(direct):
                value[direct] = (
                    cdft(2, p=pp[direct], q=qq[direct], df=dd[direct]).t
                    if which == 2
                    else -cdfnor(2, p=pp[direct], q=qq[direct]).x
                )
            at_zero, zero_q = _tails(
                np.zeros(tt.shape) if which == 2 else tt,
                dd,
                nn if which == 2 else np.zeros(nn.shape),
            )
            target = np.minimum(pp, qq)
            zero_tail = np.where(pp <= qq, at_zero, zero_q)
            zero = np.abs(zero_tail - target) <= 32 * np.finfo(float).eps * target + tolerance
            active = ~direct & ~zero
            if np.any(active):
                x, d, n = tt[active], dd[active], nn[active]
                lower = (pp <= qq)[active]
                sign = np.where(pp[active] > at_zero[active], 1.0, -1.0)
                if which == 4:
                    sign = -sign
                initial = np.maximum(np.abs(n if which == 2 else x), 1.0)
                if which == 2:
                    guess = nct.isf(qq[active], d, n)
                    initial = np.where(np.isfinite(guess) & (guess * sign > 0), abs(guess), initial)

                def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                    lp, uq = _tails(
                        sign[indices] * v if which == 2 else x[indices],
                        d[indices],
                        n[indices] if which == 2 else sign[indices] * v,
                    )
                    return np.where(lower[indices], lp, uq)

                magnitude = _invert_positive(
                    target[active],
                    np.full(d.shape, np.nextafter(0.0, 1.0)),
                    np.full(d.shape, 1e100 if which == 2 else 1e4),
                    evaluate,
                    initial=initial,
                    probability_atol=tolerance[active],
                    unbracketed_message="noncentral t solution lies outside its search bounds",
                )
                value[active] = sign * magnitude
            bound = 1e100 if which == 2 else 1e4
            value = np.where(
                (abs(value) > bound) & (abs(value) <= bound * (1 + 8 * np.finfo(float).eps)),
                np.copysign(bound, value),
                value,
            )
            if np.any(~np.isfinite(value) | (abs(value) > bound)):
                raise ValueError("noncentral t solution lies outside its search bounds")
            if which == 2:
                tt = value
            else:
                nn = value
        lp, uq = _tails(tt, dd, nn)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + tolerance + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy noncentral t inversion failed forward verification")
    return DCDFLIBNoncentralT(int(which), *map(_freeze, (pp, qq, tt, dd, nn)))


def cumtnc(t: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired legacy noncentral t tails, with signed normal-mean noncentrality."""
    result = cdftnc(t=t, df=df, pnonc=pnonc)
    return result.p, result.q
