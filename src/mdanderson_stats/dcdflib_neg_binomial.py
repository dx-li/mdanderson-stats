"""Legacy negative-binomial tails and continuous count/probability inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray
from .cdflib_beta import _quantiles
from .cdflib_beta import _tails as _beta_tails
from .dcdflib_gamma import _scaled_quantile
from .dcdflib_gamma import _tails as _gamma_tails
from .dcdflib_poisson import _nonnegative


def _log_coordinate(x: FloatArray, cx: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        return np.where(x <= cx, np.log(x), np.log1p(-cx))


def _tails(
    f: FloatArray, s: FloatArray, x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    p, q = np.ones(f.shape), np.zeros(f.shape)
    positive = s > 0
    left = positive & (x == 0)
    p[left], q[left] = 0, 1
    active = positive & (x > 0) & (y > 0)
    # Exact reductions avoid extreme beta shapes and preserve tiny complements.
    power = active & (f == 0)
    geometric = active & ~power & (s == 1)
    symmetric = active & ~power & ~geometric & (s == f + 1) & (x == y)
    p[symmetric], q[symmetric] = 0.5, 0.5
    with np.errstate(over="ignore"):
        logp = s[power] * _log_coordinate(x[power], y[power])
        logq = (f[geometric] + 1) * _log_coordinate(y[geometric], x[geometric])
    p[power], q[power] = np.exp(logp), -np.expm1(logp)
    p[geometric], q[geometric] = -np.expm1(logq), np.exp(logq)
    regular = active & ~power & ~geometric & ~symmetric
    b = f + 1
    # Gamma limits for extremely disparate beta shapes. Both shape-squared
    # and scaled-coordinate-squared corrections must be below 1e-14.
    with np.errstate(divide="ignore"):
        left = (
            regular
            & (b >= 1e20)
            & (2 * np.log(s) - np.log(b) < np.log(1e-14))
            & (np.log(b) + 2 * np.log(x) < np.log(1e-14))
        )
        right = (
            regular
            & (s >= 1e20)
            & (2 * np.log(b) - np.log(s) < np.log(1e-14))
            & (np.log(s) + 2 * np.log(y) < np.log(1e-14))
        )
    if np.any(left):
        p[left], q[left] = _gamma_tails(x[left], s[left], b[left])
    if np.any(right):
        q[right], p[right] = _gamma_tails(y[right], b[right], s[right])
    regular &= ~left & ~right
    tiny = regular & (s < 1e-300)
    if np.any(tiny):
        # Q(a,b,x) is linear in a at these scales. Evaluate at a normal
        # reference shape with a relative first-order correction below 1e-14.
        reference = 1e-14 / (1 + np.abs(np.log(x[tiny])) + np.log(b[tiny]))
        _, upper = _beta_tails(x[tiny], y[tiny], reference, b[tiny])
        q[tiny] = (upper / reference) * s[tiny]
        p[tiny] = 1 - q[tiny]
    regular &= ~tiny
    p[regular], q[regular] = _beta_tails(x[regular], y[regular], s[regular], b[regular])
    return p, q


def _tiny_chance(
    p: FloatArray, q: FloatArray, f: FloatArray, s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    _, middle = _tails(f, s, np.full(p.shape, 0.5), np.full(p.shape, 0.5))
    left = q >= middle

    def evaluate(v: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
        x, y = np.where(left[indices], v, 1 - v), np.where(left[indices], 1 - v, v)
        return _tails(f[indices], s[indices], x, y)[1]

    with np.errstate(over="ignore", divide="ignore"):
        logx = -q / s - np.log(f + 1)
        logy = (np.log(q) - np.log(s) + np.log(f + 1)) / (f + 1)
    initial = np.exp(np.maximum(np.where(left, logx, logy), np.log(np.nextafter(0.0, 1.0))))
    v = _invert_positive(
        q,
        np.full(p.shape, np.nextafter(0.0, 1.0)),
        np.full(p.shape, 0.5),
        evaluate,
        initial=initial,
        unbracketed_message="success probability is not representable",
    )
    return np.where(left, v, 1 - v), np.where(left, 1 - v, v)


@dataclass(frozen=True)
class DCDFLIBNegativeBinomial:
    """Owned immutable paired probabilities and failure/success counts."""

    which: int
    p: FloatArray
    q: FloatArray
    f: FloatArray
    s: FloatArray
    pr: FloatArray
    cpr: FloatArray


def cdfnbn(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    f: ArrayLike | None = None,
    s: ArrayLike | None = None,
    pr: ArrayLike | None = None,
    cpr: ArrayLike | None = None,
) -> DCDFLIBNegativeBinomial:
    """Compute 1=p/q, 2=failures, 3=successes or 4=pr/cpr.

    Finite counts are nonnegative; computed counts are bounded by 1e100.
    Zero required successes give p=1, including at pr=0, as in the F95 Python API.
    Inverse q must be positive. Counts are real, without integer rounding.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (f,), 3: (s,), 4: (pr, cpr)}[which]):
        raise ValueError("omit the parameter group being computed")
    ff = _nonnegative(f, "f") if which != 2 else np.asarray(0.0)
    ss = _nonnegative(s, "s") if which != 3 else np.asarray(1.0)
    xx, yy = _probability_pair(pr, cpr) if which != 4 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, ff, ss, xx, yy = np.broadcast_arrays(pp, qq, ff, ss, xx, yy)
    if which == 1:
        pp, qq = _tails(ff, ss, xx, yy)
    else:
        if np.any(qq <= 0):
            raise ValueError("legacy negative-binomial inversion requires positive q")
        if which == 4:
            if np.any(ss == 0):
                raise ValueError("zero successes do not identify a success probability")
            power, geometric = ff == 0, (ss == 1) & (ff != 0)
            regular = ~power & ~geometric
            xx, yy = np.zeros(pp.shape), np.ones(pp.shape)
            regular &= pp > 0
            b = ff + 1
            with np.errstate(divide="ignore"):
                left = regular & (b >= 1e20) & (2 * np.log(ss) - np.log(b) < np.log(1e-14))
                right = regular & (ss >= 1e20) & (2 * np.log(b) - np.log(ss) < np.log(1e-14))
            xx[left] = _scaled_quantile(pp[left], qq[left], ss[left], b[left])
            yy[left] = 1 - xx[left]
            yy[right] = _scaled_quantile(qq[right], pp[right], b[right], ss[right])
            xx[right] = 1 - yy[right]
            regular &= ~left & ~right
            tiny = regular & (ss < 1e-300)
            if np.any(tiny):
                xx[tiny], yy[tiny] = _tiny_chance(pp[tiny], qq[tiny], ff[tiny], ss[tiny])
            regular &= ~tiny
            xx[regular], yy[regular] = _quantiles(pp[regular], qq[regular], ss[regular], b[regular])
            logp = _log_coordinate(pp, qq)
            logq = _log_coordinate(qq, pp)
            with np.errstate(over="ignore"):
                z = logp[power] / ss[power]
            xx[power], yy[power] = np.exp(z), -np.expm1(z)
            z = logq[geometric] / (ff[geometric] + 1)
            xx[geometric], yy[geometric] = -np.expm1(z), np.exp(z)
        else:
            if np.any((pp <= 0) | (xx <= 0) | (yy <= 0)) or (which == 2 and np.any(ss == 0)):
                raise ValueError("count inversion requires positive tails and interior pr/cpr")
            fixed_f, fixed_s, x, y = ff.ravel(), ss.ravel(), xx.ravel(), yy.ravel()
            lower = (pp <= qq).ravel()
            solve_s = which == 3

            def evaluate(value: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(
                    fixed_f[indices] if solve_s else value - 1,
                    value if solve_s else fixed_s[indices],
                    x[indices],
                    y[indices],
                )
                return np.where(lower[indices], lp, uq)

            # Mean-based anchors preserve very large central roots. For small
            # upper tails, success counts can be below 1e-300.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                factor = np.minimum(1.0, 2 * qq)
                initial = (ff + 1) * (xx / yy) * factor if solve_s else ss * (yy / xx) + 1
                log_initial = (
                    np.log(ff + 1) + np.log(xx) - np.log(yy) + np.log(factor)
                    if solve_s
                    else np.log(ss) + np.log(yy) - np.log(xx)
                )
            fallback = np.exp(np.clip(log_initial, np.log(np.nextafter(0.0, 1.0)), np.log(1e100)))
            initial = np.where(np.isfinite(initial) & (initial > 0), initial, fallback)
            with np.errstate(over="ignore"):
                closed = (
                    _log_coordinate(pp, qq) / _log_coordinate(xx, yy)
                    if solve_s
                    else _log_coordinate(qq, pp) / _log_coordinate(yy, xx)
                )
            initial = np.where((ff == 0) if solve_s else (ss == 1), closed, initial)
            value = _invert_positive(
                np.minimum(pp, qq),
                np.full(pp.shape, np.nextafter(0.0, 1.0) if solve_s else 1.0),
                np.full(pp.shape, 1e100),
                evaluate,
                initial=initial,
                unbracketed_message="negative-binomial count lies outside [0,1e100]",
            )
            if solve_s:
                ss = value
            else:
                ff = value - 1
        if np.any(
            ~np.isfinite(xx)
            | ~np.isfinite(yy)
            | (xx < 0)
            | (xx > 1)
            | (yy <= 0)
            | (yy > 1)
            | ((pp > 0) & (xx == 0))
        ):
            raise ValueError("success probability pair is not representable in its required domain")
        lp, uq = _tails(ff, ss, xx, yy)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy negative-binomial inverse failed forward verification")
    return DCDFLIBNegativeBinomial(int(which), *map(_freeze, (pp, qq, ff, ss, xx, yy)))


def cumnbn(
    f: ArrayLike, s: ArrayLike, pr: ArrayLike | None = None, *, cpr: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Paired negative-binomial tails; source S/XN correspond to f/s."""
    r = cdfnbn(f=f, s=s, pr=pr, cpr=cpr)
    return r.p, r.q
