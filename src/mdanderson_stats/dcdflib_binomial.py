"""Legacy DCDFLIB binomial contracts with distinct success/trial search bounds."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray
from .dcdflib_neg_binomial import _log_coordinate, cdfnbn
from .dcdflib_neg_binomial import _tails as _nb_tails
from .dcdflib_poisson import _nonnegative


def _tails(
    s: FloatArray, n: FloatArray, x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    # I_(1-pr)(n-s,s+1), including P=1 when the first shape vanishes.
    return _nb_tails(s, n - s, y, x)


def _success(
    p: FloatArray, q: FloatArray, n: FloatArray, x: FloatArray, y: FloatArray
) -> FloatArray:
    lp, uq = _tails(np.zeros(p.shape), n, x, y)
    lower = p <= q
    edge = np.where(lower, lp, uq)
    target = np.minimum(p, q)
    target = np.where(np.abs(target - edge) <= 32 * np.finfo(float).eps * edge, edge, target)
    if np.any(np.where(lower, target < edge, target > edge)):
        raise ValueError("binomial success solution lies outside [0,n]")
    active = target != edge
    result = np.zeros(p.shape)
    if np.any(active):
        nn, xx, yy, left = n[active], x[active], y[active], lower[active]

        def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
            a, b = _tails(v, nn[indices], xx[indices], yy[indices])
            return np.where(left[indices], a, b)

        result[active] = _invert_positive(
            target[active],
            np.full(nn.shape, np.nextafter(0.0, 1.0)),
            nn,
            evaluate,
            initial=np.maximum(nn * xx, np.nextafter(0.0, 1.0)),
            unbracketed_message="binomial success solution is not representable in [0,n]",
        )
    return result


@dataclass(frozen=True)
class DCDFLIBBinomial:
    """Computed group and six owned immutable broadcast arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    s: FloatArray
    n: FloatArray
    pr: FloatArray
    cpr: FloatArray


def cdfbin(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    s: ArrayLike | None = None,
    n: ArrayLike | None = None,
    pr: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> DCDFLIBBinomial:
    """Compute 1=p/q, 2=successes, 3=trials or 4=pr/cpr; omit that group.

    Input n is positive finite and 0<=s<=n. Computed s spans [0,n], including
    n>1e100. Computed n lies in [1e-100,1e100] and must not be below s.
    Counts are continuous real values, without integer quantile rounding.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (s,), 3: (n,), 4: (pr, cpr)}[which]):
        raise ValueError("omit the parameter group being computed")
    ss = _nonnegative(s, "s") if which != 2 else np.asarray(0.0)
    nn = _nonnegative(n, "n") if which != 3 else np.asarray(1.0)
    if which != 3 and np.any(nn <= 0):
        raise ValueError("n must be positive")
    xx, yy = _probability_pair(pr, cpr) if which != 4 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, ss, nn, xx, yy = np.broadcast_arrays(pp, qq, ss, nn, xx, yy)
    if which in (1, 4) and np.any(ss > nn):
        raise ValueError("s must not exceed n")
    if which == 1:
        pp, qq = _tails(ss, nn, xx, yy)
    else:
        if which == 4:
            if np.any(ss == nn):
                raise ValueError("s=n does not identify a success probability")
            interior = (pp > 0) & (qq > 0)
            xx, yy = np.where(pp == 0, 1.0, 0.0), np.where(pp == 0, 0.0, 1.0)
            if np.any(interior):
                r = cdfnbn(
                    4, p=pp[interior], q=qq[interior], f=ss[interior], s=nn[interior] - ss[interior]
                )
                xx[interior], yy[interior] = r.cpr, r.pr
        else:
            if np.any((pp <= 0) | (xx <= 0) | ((yy <= 0) & (qq > 0))):
                raise ValueError(
                    "count inversion is unattainable or does not identify a unique count"
                )
            endpoint = qq == 0
            if which == 3 and np.any((ss > 1e100) | (endpoint & (ss < 1e-100))):
                raise ValueError("binomial trial solution lies outside [1e-100,1e100]")
            if which == 2:
                answer = nn.copy()
                active = ~endpoint
                answer[active] = _success(
                    pp[active], qq[active], nn[active], xx[active], yy[active]
                )
                ss = answer
            else:
                answer = ss.copy()
                active = ~endpoint
                if np.any(active):
                    fixed, x, y, left = ss[active], xx[active], yy[active], (pp <= qq)[active]

                    def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                        a, b = _tails(fixed[indices], v, x[indices], y[indices])
                        return np.where(left[indices], a, b)

                    with np.errstate(over="ignore"):
                        initial = (fixed + 1) / x
                        exact = _log_coordinate(pp[active], qq[active]) / _log_coordinate(y, x)
                    initial = np.where(fixed == 0, exact, initial)
                    answer[active] = _invert_positive(
                        np.minimum(pp[active], qq[active]),
                        np.maximum(fixed, 1e-100),
                        np.full(fixed.shape, 1e100),
                        evaluate,
                        initial=initial,
                        unbracketed_message=(
                            "binomial trial solution lies outside [max(s,1e-100),1e100]"
                        ),
                    )
                nn = answer
        lp, uq = _tails(ss, nn, xx, yy)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy binomial inverse failed forward verification")
    return DCDFLIBBinomial(int(which), *map(_freeze, (pp, qq, ss, nn, xx, yy)))


def cumbin(
    s: ArrayLike, n: ArrayLike, pr: ArrayLike | None = None, *, cpr: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Paired binomial tails with real success and trial counts."""
    r = cdfbin(s=s, n=n, pr=pr, cpr=cpr)
    return r.p, r.q
