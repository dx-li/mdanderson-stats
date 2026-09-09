"""Incomplete-beta series support with explicit source domains."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_beta import _tails
from .cdflib_beta_factors import _positive_parts
from .cdflib_beta_shift import _normalized


def _strict_product_bound(b: FloatArray, eps: FloatArray, a: FloatArray) -> NDArray[np.bool_]:
    # Division avoids overflowing/underflowing eps*a. Resolve a rounded tie
    # with exact integer ratios, rather than changing the strict source bound.
    with np.errstate(over="ignore", under="ignore"):
        ratio = b / a
    accepted = ratio < eps
    for i in np.flatnonzero(ratio == eps):
        bn, bd = float(b[i]).as_integer_ratio()
        en, ed = float(eps[i]).as_integer_ratio()
        an, ad = float(a[i]).as_integer_ratio()
        accepted[i] = bn * ed * ad < en * an * bd
    return accepted


def _lower_series(a: FloatArray, b: FloatArray, x: FloatArray, eps: FloatArray) -> FloatArray:
    term, series = np.ones(a.shape), np.zeros(a.shape)
    active = np.ones(a.shape, dtype=bool)
    for n in range(1, 129):
        term[active] *= ((n - b[active]) / n) * x[active]
        add = term[active] * (a[active] / (a[active] + n))
        series[active] += add
        # Positive successive integral terms have ratio at most x <= 1/2.
        bound = add * x[active] / (1 - x[active])
        active[active] = bound > eps[active] * (1 + series[active])
        if not np.any(active):
            break
    else:
        raise ArithmeticError("beta power series did not meet its remainder bound")
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
    if np.any((bb >= ee) | ~_strict_product_bound(bb, ee, aa)):
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
