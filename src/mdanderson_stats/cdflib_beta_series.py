"""Incomplete-beta series support with explicit source domains."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_beta import _tails
from .cdflib_beta_factors import _positive_parts
from .cdflib_beta_shift import _normalized
from .cdflib_beta_support import _small_binomial
from .cdflib_gamma_support import psi
from .cdflib_incomplete_gamma import gratio


def _product_bound(
    b: FloatArray, eps: FloatArray, a: FloatArray, *, inclusive: bool = False
) -> NDArray[np.bool_]:
    # Division avoids overflowing/underflowing eps*a. Resolve a rounded tie
    # with exact integer ratios, preserving the source comparison.
    with np.errstate(over="ignore", under="ignore"):
        ratio = b / a
    accepted = ratio <= eps if inclusive else ratio < eps
    for i in np.flatnonzero(ratio == eps):
        bn, bd = float(b[i]).as_integer_ratio()
        en, ed = float(eps[i]).as_integer_ratio()
        an, ad = float(a[i]).as_integer_ratio()
        left, right = bn * ed * ad, en * an * bd
        accepted[i] = left <= right if inclusive else left < right
    return accepted


def _integral_series(
    a: FloatArray, b: FloatArray, x: FloatArray, eps: FloatArray, *, divided: bool = False
) -> FloatArray:
    """Return a*H, or H when divided, for the beta integral power correction."""
    term, series = np.ones(a.shape), np.zeros(a.shape)
    active = np.ones(a.shape, dtype=bool)
    for n in range(1, 129):
        term[active] *= ((n - b[active]) / n) * x[active]
        numerator = 1 if divided else a[active]
        add = term[active] * (numerator / (a[active] + n))
        series[active] += add
        # Positive successive integral terms have ratio at most x <= 1/2.
        bound = add * x[active] / (1 - x[active])
        active[active] = bound > eps[active] * (1 + series[active])
        if not np.any(active):
            break
    else:
        raise ArithmeticError("beta power series did not meet its remainder bound")
    return series


def _lower_series(a: FloatArray, b: FloatArray, x: FloatArray, eps: FloatArray) -> FloatArray:
    series = _integral_series(a, b, x, eps)
    prefactor, exponent, divisor = _positive_parts(a, b, x, 1 - x, np.zeros(a.shape))
    # Remove y**b and divide by a only after retaining the complete scale.
    return _normalized(prefactor, exponent - b * np.log1p(-x), divisor, a, np.log1p(series))


def fpser(a: ArrayLike, b: ArrayLike, x: ArrayLike, eps: ArrayLike = 5e-15) -> FloatArray:
    """Compute I_x(a,b) for b < min(eps,eps*a) and 0 <= x <= 1/2.

    Shapes and eps must be finite and positive. Inputs broadcast to immutable
    float64 outputs. The full beta normalization is retained even for loose
    tolerances; truncation tolerance is capped at 5e-15 and floored at four ulps
    at one. Endpoint and representable subnormal probabilities are preserved.
    """
    aa, bb, xx, ee = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), finite(x, "x"), finite(eps, "eps")
    )
    shape = aa.shape
    aa, bb, xx, ee = (v.ravel() for v in (aa, bb, xx, ee))
    if np.any((aa <= 0) | (bb <= 0) | (ee <= 0) | (xx < 0) | (xx > 0.5)):
        raise ValueError("positive a, b and eps and 0 <= x <= 1/2 are required")
    if np.any((bb >= ee) | ~_product_bound(bb, ee, aa)):
        raise ValueError("fpser requires b < min(eps,eps*a)")
    tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(ee, 5e-15))
    result = np.zeros(aa.shape)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        unit_a = (xx > 0) & (aa == 1)
        result[unit_a] = -np.expm1(bb[unit_a] * np.log1p(-xx[unit_a]))
        unit_b = (xx > 0) & (bb == 1) & ~unit_a
        result[unit_b] = np.exp(aa[unit_b] * np.log(xx[unit_b]))
        active = (xx > 0) & ~unit_a & ~unit_b
        positive = active & (bb < 1)
        if np.any(positive):
            result[positive] = _lower_series(
                aa[positive], bb[positive], xx[positive], tolerance[positive]
            )
        # Very loose eps can admit b>1 under the literal source inequality.
        # Evaluate the beta integral with the existing compiled tail kernel
        # instead of applying a tiny-b approximation outside its useful range.
        general = active & ~positive
        if np.any(general):
            result[general] = _tails(xx[general], 1 - xx[general], aa[general], bb[general])[0]
    if np.any(~np.isfinite(result) | (result < 0) | (result > 1)):
        raise ArithmeticError("beta series produced an invalid probability")
    return _freeze(result.reshape(shape))


def _small_upper(a: FloatArray, b: FloatArray, x: FloatArray, eps: FloatArray) -> FloatArray:
    h = _integral_series(a, b, x, eps, divided=True)
    # Gamma recurrences separate b/(a+b) from a correction factored in a*b.
    # Keeping log(P) in these small pieces retains Q even when P rounds to 1.
    logp = -np.log1p(a / b) + a * np.log(x) + _small_binomial(a, b, a + b) + np.log1p(a * h)
    return -np.expm1(logp)


def _tiny_upper(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    bx = b * x
    term = x - bx
    series = term.copy()
    # For x<=1/2 and bx<=1, successive absolute terms shrink by at least
    # a factor of two from n=2 onward. Sixty-four terms suffice in float64.
    for n in range(2, 65):
        term *= x - bx / n
        series += term / n
    return -a * (np.log(x) + psi(b) + np.euler_gamma + series)


def apser(a: ArrayLike, b: ArrayLike, x: ArrayLike, eps: ArrayLike = 5e-15) -> FloatArray:
    """Compute I_(1-x)(b,a) for a<=min(eps,eps*b), b*x<=1 and x<=1/2.

    Shapes and eps must be finite and positive; x must be nonnegative.
    Small upper tails are evaluated directly. Results broadcast to immutable
    float64 arrays. eps controls the source domain and series tolerance.
    """
    aa, bb, xx, ee = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), finite(x, "x"), finite(eps, "eps")
    )
    shape = aa.shape
    aa, bb, xx, ee = (v.ravel() for v in (aa, bb, xx, ee))
    if np.any((aa <= 0) | (bb <= 0) | (ee <= 0) | (xx < 0) | (xx > 0.5)):
        raise ValueError("positive a, b and eps and 0 <= x <= 1/2 are required")
    positive = xx > 0
    if np.any((aa > ee) | ~_product_bound(aa, ee, bb, inclusive=True)) or np.any(
        _product_bound(np.ones(np.count_nonzero(positive)), bb[positive], xx[positive])
    ):
        raise ValueError("apser requires a<=min(eps,eps*b) and b*x<=1")
    tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(ee, 5e-15))
    result = np.ones(aa.shape)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        unit_b = positive & (bb == 1)
        result[unit_b] = -np.expm1(aa[unit_b] * np.log(xx[unit_b]))
        unit_a = positive & (aa == 1) & ~unit_b
        result[unit_a] = np.exp(bb[unit_a] * np.log1p(-xx[unit_a]))
        active = positive & ~unit_b & ~unit_a
        small = active & (aa + bb <= 0.25)
        if np.any(small):
            result[small] = _small_upper(aa[small], bb[small], xx[small], tolerance[small])
        large = active & ~small & (bb >= 1e15)
        if np.any(large):
            result[large] = gratio(aa[large], bb[large] * xx[large])[1]
        tiny = active & ~small & ~large & (aa <= 1e-18)
        if np.any(tiny):
            result[tiny] = _tiny_upper(aa[tiny], bb[tiny], xx[tiny])
        regular = active & ~small & ~large & ~tiny
        if np.any(regular):
            result[regular] = _tails(xx[regular], 1 - xx[regular], aa[regular], bb[regular])[1]
    if np.any(~np.isfinite(result) | (result < 0) | (result > 1)):
        raise ArithmeticError("beta upper-tail evaluation produced an invalid probability")
    return _freeze(result.reshape(shape))
