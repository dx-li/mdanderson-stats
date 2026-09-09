"""Legacy noncentral chi-square contracts with checked wide-domain tails."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ncx2

from ._cdflib import _freeze
from ._dcdflib import _invert_positive
from ._validation import FloatArray, finite
from .dcdflib_chisq import _quantile as _central_quantile
from .dcdflib_chisq import _tails as _central_tails
from .dcdflib_f import _positive
from .dcdflib_normal import cumnor
from .dcdflib_poisson import _nonnegative


def _tails(x: FloatArray, df: FloatArray, nc: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = np.zeros(x.shape), np.ones(x.shape)
    active = x > 0
    central = active & (nc == 0)
    if np.any(central):
        p[central], q[central] = _central_tails(x[central], df[central])
    active &= ~central
    # Chernoff bounds from the noncentral chi-square Laplace/MGF transforms.
    # Only replace a tail when its upper bound is below half a subnormal ulp.
    cutoff = np.log(np.nextafter(0.0, 1.0)) - np.log(2.0)
    lower_zero = active & (x / 2 - (df / 2) * np.log(2.0) - nc / 4 < cutoff)
    upper_zero = active & (-x / 4 + (df / 2) * np.log(2.0) + nc / 2 < cutoff)
    p[upper_zero], q[upper_zero] = 1, 0
    active &= ~lower_zero & ~upper_zero
    tiny = active & (x < 1e-80)
    if np.any(tiny):
        # Higher Poisson terms relative to j=0 are bounded by a series in
        # nc*x/(2*(df+2)); after Chernoff elimination nc is modest here.
        lp, uq = _central_tails(x[tiny], df[tiny])
        weight = np.exp(-nc[tiny] / 2)
        p[tiny] = weight * lp
        q[tiny] = -np.expm1(-nc[tiny] / 2) + weight * uq
    active &= ~tiny
    small_nc = active & (nc <= 1e-12)
    if np.any(small_nc):
        xx, dd, mean = x[small_nc], df[small_nc], nc[small_nc] / 2
        weight = np.exp(-mean)
        lp, uq = _central_tails(xx, dd)
        lower, upper = weight * lp, weight * uq
        # Preserve both the central and noncentral contributions, including
        # tiny df where a compiled upper tail drops the j=0 term.
        for j in range(1, 4):
            weight = weight * mean / j
            lp, uq = _central_tails(xx, dd + 2 * j)
            lower += weight * lp
            upper += weight * uq
        p[small_nc], q[small_nc] = lower, upper
    active &= ~small_nc
    normal = active & (df == 1)
    if np.any(normal):
        xx, nn = x[normal], nc[normal]
        root, shift = np.sqrt(xx), np.sqrt(nn)
        total = root + shift
        delta = (xx - nn) / total
        lp, uq = cumnor(delta)
        other, _ = cumnor(-total)
        p[normal], q[normal] = lp - other, uq + other
        # A local integral avoids cancellation between nearby normal CDFs.
        local = root * (shift + 1) < 1e-4
        if np.any(local):
            logp = (
                np.log(2.0)
                + 0.5 * np.log(xx[local])
                - nn[local] / 2
                - 0.5 * np.log(2 * np.pi)
                + np.log1p((nn[local] - 1) * xx[local] / 6)
            )
            values = np.exp(logp)
            indices = np.flatnonzero(normal)
            p.flat[indices[local]], q.flat[indices[local]] = values, 1 - values
    active &= ~normal
    if np.any(active):
        p[active] = ncx2.cdf(x[active], df[active], nc[active])
        q[active] = ncx2.sf(x[active], df[active], nc[active])
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ArithmeticError("noncentral chi-square tail evaluation failed")
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)


@dataclass(frozen=True)
class DCDFLIBNoncentralChiSquare:
    """Computed group and owned immutable arrays; inverse q is 1-p."""

    which: int
    p: FloatArray
    q: FloatArray
    x: FloatArray
    df: FloatArray
    pnonc: FloatArray


def cdfchn(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    x: ArrayLike | None = None,
    df: ArrayLike | None = None,
    pnonc: ArrayLike | None = None,
) -> DCDFLIBNoncentralChiSquare:
    """Compute 1=p/q, 2=x, 3=df or 4=pnonc; omit the computed group.

    Inverse p is required in [0,1-1e-16]; q is ignored, including its shape.
    Inputs x/pnonc are nonnegative finite and df positive finite. Computed
    x is in [0,1e100], df [1e-100,1e100], and pnonc [0,1e4].
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (x,), 3: (df,), 4: (pnonc,)}[which]):
        raise ValueError("omit the parameter group being computed")
    xx = _nonnegative(x, "x") if which != 2 else np.asarray(0.0)
    dd = _positive(df, "df") if which != 3 else np.asarray(1.0)
    nn = _nonnegative(pnonc, "pnonc") if which != 4 else np.asarray(0.0)
    if which == 1:
        pp = np.asarray(0.5)
    else:
        if p is None:
            raise ValueError("p is required; the legacy interface ignores q")
        pp = finite(p, "p")
        if np.any((pp < 0) | (pp > 1 - 1e-16)):
            raise ValueError("inverse p must be in [0,1-1e-16]")
    pp, xx, dd, nn = np.broadcast_arrays(pp, xx, dd, nn)
    qq = 1 - pp
    tolerance = np.spacing(pp) / 2
    if which == 1:
        pp, qq = _tails(xx, dd, nn)
    elif which == 2:
        xx = np.zeros(pp.shape)
        central = (nn == 0) & (pp > 0)
        if np.any(central):
            xx[central] = _central_quantile(pp[central], qq[central], dd[central])
        active = (nn > 0) & (pp > 0)
        if np.any(active):
            d, n, left = dd[active], nn[active], (pp <= qq)[active]
            initial = np.empty(d.shape)
            initial[left] = ncx2.ppf(pp[active][left], d[left], n[left])
            initial[~left] = ncx2.isf(qq[active][~left], d[~left], n[~left])
            with np.errstate(over="ignore"):
                fallback = np.minimum(d + n, 1e100)
            initial = np.where(np.isfinite(initial) & (initial > 0), initial, fallback)

            def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(v, d[indices], n[indices])
                return np.where(left[indices], lp, uq)

            xx[active] = _invert_positive(
                np.minimum(pp[active], qq[active]),
                np.full(d.shape, np.nextafter(0.0, 1.0)),
                np.full(d.shape, 1e100),
                evaluate,
                initial=initial,
                probability_atol=tolerance[active],
                unbracketed_message="noncentral chi-square x is not representable in [0,1e100]",
            )
        xx = np.where((xx > 1e100) & (xx <= 1e100 * (1 + 8 * np.finfo(float).eps)), 1e100, xx)
        if np.any(~np.isfinite(xx) | (xx < 0) | (xx > 1e100) | ((xx == 0) & (pp > 0))):
            raise ValueError("noncentral chi-square x is not representable in [0,1e100]")
    else:
        if np.any((pp <= 0) | (xx <= 0)):
            raise ValueError("parameter inversion requires positive x and p")
        fixed_x, fixed_df, fixed_nc = xx.ravel(), dd.ravel(), nn.ravel()
        left = (pp <= qq).ravel()

        def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
            lp, uq = _tails(
                fixed_x[indices],
                v if which == 3 else fixed_df[indices],
                fixed_nc[indices] if which == 3 else v - 1,
            )
            return np.where(left[indices], lp, uq)

        initial = np.maximum(xx - nn, 1.0) if which == 3 else np.maximum(xx - dd, 0.0) + 1
        value = _invert_positive(
            np.minimum(pp, qq),
            np.full(pp.shape, 1e-100 if which == 3 else 1.0),
            np.full(pp.shape, 1e100 if which == 3 else 10001.0),
            evaluate,
            initial=initial,
            probability_atol=tolerance,
            unbracketed_message="noncentral chi-square parameter lies outside its search bounds",
        )
        if which == 3:
            dd = value
        else:
            nn = value - 1
    if which != 1:
        lp, uq = _tails(xx, dd, nn)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + tolerance + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError(
                "legacy noncentral chi-square inverse failed forward verification"
            )
    return DCDFLIBNoncentralChiSquare(int(which), *map(_freeze, (pp, qq, xx, dd, nn)))


def cumchn(x: ArrayLike, df: ArrayLike, pnonc: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired noncentral chi-square tails over the legacy finite input domain."""
    r = cdfchn(x=x, df=df, pnonc=pnonc)
    return r.p, r.q
