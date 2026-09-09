"""Legacy beta contracts with wide shapes and complementary inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray
from .cdflib_beta import _quantiles
from .dcdflib_neg_binomial import _log_coordinate, cdfnbn
from .dcdflib_neg_binomial import _tails as _nb_tails
from .dcdflib_poisson import _nonnegative


def _tails(
    x: FloatArray, y: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray]:
    p, q = np.zeros(x.shape), np.ones(x.shape)
    p[y == 0], q[y == 0] = 1, 0
    active = (x > 0) & (y > 0)
    left = active & (b >= 1)
    right = active & ~left & (a >= 1)
    if np.any(left):
        p[left], q[left] = _nb_tails(b[left] - 1, a[left], x[left], y[left])
    if np.any(right):
        q[right], p[right] = _nb_tails(a[right] - 1, b[right], y[right], x[right])
    small = active & ~left & ~right
    if np.any(small):
        aa, bb, xx, yy = a[small], b[small], x[small], y[small]
        # I_x(a,b)=I_x(a+1,b)+x^a*y^b/(a*B(a,b)).
        # Apply the reflected recurrence independently for Q: both sums
        # are positive, retaining tiny tails when either shape is subnormal.
        _, lp = _nb_tails(aa, bb, yy, xx)
        _, uq = _nb_tails(bb, aa, xx, yy)
        total = aa + bb
        log_factor = (
            aa * _log_coordinate(xx, yy)
            + bb * _log_coordinate(yy, xx)
            + gammaln(1 + total)
            - gammaln(1 + aa)
            - gammaln(1 + bb)
        )
        factor = np.exp(log_factor)
        p[small], q[small] = lp + (bb / total) * factor, uq + (aa / total) * factor
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (q < 0)):
        raise ArithmeticError("legacy beta tail evaluation failed")
    smaller = np.minimum(p, q)
    if np.any(smaller > 0.5 + 8 * np.finfo(float).eps):
        raise ArithmeticError("legacy beta tails are inconsistent")
    symmetric = (a == b) & (x == y)
    return (
        np.where(symmetric, 0.5, np.where(p <= q, p, 1 - q)),
        np.where(symmetric, 0.5, np.where(p <= q, 1 - p, q)),
    )


def _small_quantile(
    p: FloatArray, q: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray]:
    # Numerical medians may be unresolved over a large coordinate interval
    # when both shapes are tiny. Choose the midpoint when it matches.
    midp, midq = _tails(np.full(p.shape, 0.5), np.full(p.shape, 0.5), a, b)
    lower = p <= q
    target = np.minimum(p, q)
    middle = np.where(lower, midp, midq)
    done = np.abs(target - middle) <= 32 * np.finfo(float).eps * target
    x, y = np.full(p.shape, 0.5), np.full(p.shape, 0.5)
    active = ~done
    if np.any(active):
        aa, bb, ll = a[active], b[active], lower[active]
        left = np.where(ll, target[active] < middle[active], target[active] > middle[active])

        def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
            xx, yy = np.where(left[indices], v, 1 - v), np.where(left[indices], 1 - v, v)
            lp, uq = _tails(xx, yy, aa[indices], bb[indices])
            return np.where(ll[indices], lp, uq)

        # The small-coordinate power law supplies a logarithmic anchor.
        # Omitting the bounded gamma correction only shifts this starting
        # estimate; bracketing and the forward check determine the answer.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logx = _log_coordinate(p[active], q[active]) / aa + np.log1p(aa / bb) / aa
            logy = _log_coordinate(q[active], p[active]) / bb + np.log1p(bb / aa) / bb
        anchor = np.where(left, logx, logy)
        anchor = np.where(np.isnan(anchor), np.log(0.25), anchor)
        initial = np.exp(np.clip(anchor, np.log(np.nextafter(0.0, 1.0)), np.log(0.5)))
        v = _invert_positive(
            target[active],
            np.full(aa.shape, np.nextafter(0.0, 1.0)),
            np.full(aa.shape, 0.5),
            evaluate,
            initial=initial,
            unbracketed_message="beta quantile is not representable",
        )
        x[active], y[active] = np.where(left, v, 1 - v), np.where(left, 1 - v, v)
    return x, y


@dataclass(frozen=True)
class DCDFLIBBeta:
    """Computed group and six owned immutable broadcast arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    x: FloatArray
    cx: FloatArray
    a: FloatArray
    b: FloatArray


def cdfbet(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    x: ArrayLike | None = None,
    cx: ArrayLike | None = None,
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
) -> DCDFLIBBeta:
    """Compute 1=p/q, 2=x/cx, 3=a or 4=b; omit the computed group.

    Input shapes are positive finite; computed shapes lie in [1e-100,1e100].
    Either coordinate of each complementary pair may be supplied.
    Shape inversions require interior coordinates and strictly positive tails.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (x, cx), 3: (a,), 4: (b,)}[which]):
        raise ValueError("omit the parameter group being computed")
    aa = _nonnegative(a, "a") if which != 3 else np.asarray(1.0)
    bb = _nonnegative(b, "b") if which != 4 else np.asarray(1.0)
    if np.any(aa <= 0) or np.any(bb <= 0):
        raise ValueError("beta shapes must be positive")
    xx, yy = _probability_pair(x, cx) if which != 2 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, xx, yy, aa, bb = np.broadcast_arrays(pp, qq, xx, yy, aa, bb)
    if which == 1:
        pp, qq = _tails(xx, yy, aa, bb)
    elif which == 2:
        xx, yy = np.where(pp == 0, 0.0, 1.0), np.where(pp == 0, 1.0, 0.0)
        symmetric = (aa == bb) & (pp == qq)
        xx[symmetric], yy[symmetric] = 0.5, 0.5
        active = (pp > 0) & (qq > 0) & ~symmetric
        left = active & (bb >= 1)
        right = active & ~left & (aa >= 1)
        if np.any(left):
            r = cdfnbn(4, p=pp[left], q=qq[left], f=bb[left] - 1, s=aa[left])
            xx[left], yy[left] = r.pr, r.cpr
        if np.any(right):
            r = cdfnbn(4, p=qq[right], q=pp[right], f=aa[right] - 1, s=bb[right])
            yy[right], xx[right] = r.pr, r.cpr
        small = active & ~left & ~right
        tiny = small & (np.minimum(aa, bb) < 1e-300)
        regular = small & ~tiny
        if np.any(regular):
            xx[regular], yy[regular] = _quantiles(
                pp[regular], qq[regular], aa[regular], bb[regular]
            )
        if np.any(tiny):
            xx[tiny], yy[tiny] = _small_quantile(pp[tiny], qq[tiny], aa[tiny], bb[tiny])
        if np.any(active & ((xx <= 0) | (yy <= 0))):
            raise ValueError("beta quantile is not representable")
    else:
        if np.any((pp <= 0) | (qq <= 0) | (xx <= 0) | (yy <= 0)):
            raise ValueError("shape inversion requires interior coordinates and positive tails")
        solve_a = which == 3
        fixed = (bb if solve_a else aa).ravel()
        xx_flat, yy_flat, lower = xx.ravel(), yy.ravel(), (pp <= qq).ravel()

        def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
            lp, uq = _tails(
                xx_flat[indices],
                yy_flat[indices],
                v if solve_a else fixed[indices],
                fixed[indices] if solve_a else v,
            )
            return np.where(lower[indices], lp, uq)

        with np.errstate(over="ignore", divide="ignore"):
            initial = bb * (xx / yy) if solve_a else aa * (yy / xx)
            power = (
                _log_coordinate(pp, qq) / _log_coordinate(xx, yy)
                if solve_a
                else _log_coordinate(qq, pp) / _log_coordinate(yy, xx)
            )
        initial = np.where((bb if solve_a else aa) == 1, power, initial)
        result = _invert_positive(
            np.minimum(pp, qq),
            np.full(pp.shape, 1e-100),
            np.full(pp.shape, 1e100),
            evaluate,
            initial=initial,
            unbracketed_message="beta shape solution lies outside [1e-100,1e100]",
        )
        if solve_a:
            aa = result
        else:
            bb = result
    if which != 1:
        lp, uq = _tails(xx, yy, aa, bb)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy beta inverse failed forward verification")
    return DCDFLIBBeta(int(which), *map(_freeze, (pp, qq, xx, yy, aa, bb)))


def cumbet(
    x: ArrayLike | None, a: ArrayLike, b: ArrayLike, *, cx: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Paired beta tails with explicit complementary coordinates."""
    r = cdfbet(x=x, cx=cx, a=a, b=b)
    return r.p, r.q
