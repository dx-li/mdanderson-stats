"""Normal inversion and the legacy normal/t initial approximations."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_elementary import evaluate_polynomial
from .dcdflib_normal import cdfnor

_NORMAL_NUM = (-0.322232431088, -1.0, -0.342242088547, -0.0204231210245, -0.0000453642210148)
_NORMAL_DEN = (0.0993484626060, 0.588581570495, 0.531103462366, 0.103537752850, 0.0038560700634)
_T_COEFFICIENTS = (
    (1.0, 1.0),
    (3.0, 16.0, 5.0),
    (-15.0, 17.0, 19.0, 3.0),
    (-945.0, -1920.0, 1482.0, 776.0, 79.0),
)
_T_DENOMINATORS = (4.0, 96.0, 384.0, 92160.0)


def stvaln(p: ArrayLike) -> FloatArray:
    """Archived rational normal starting value for 0<p<1, not a refined quantile.

    Retains the historical approximation, including its small nonzero median.
    Use dinvnr for a refined inverse and explicit complementary probabilities.
    """
    probability = finite(p, "p")
    if np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("stvaln requires 0<p<1")
    small = np.where(probability <= 0.5, probability, 1 - probability)
    y = np.sqrt(-2 * np.log(small))
    value = y + evaluate_polynomial(_NORMAL_NUM, y) / evaluate_polynomial(_NORMAL_DEN, y)
    return _freeze(np.where(probability <= 0.5, -value, value))


def dinvnr(p: ArrayLike | None, q: ArrayLike | None = None) -> FloatArray:
    """Refined normal inverse, preserving the smaller complementary probability.

    Either tail may be omitted; both must be positive after completion. Reuses
    the independently validated legacy normal inverse and its forward check.
    """
    # Forward verification deliberately permits subnormal probabilities.
    with np.errstate(under="ignore"):
        return cdfnor(2, p=p, q=q).x


def dt1(p: ArrayLike | None, q: ArrayLike | None, df: ArrayLike) -> FloatArray:
    """Four-term inverse-df t starting approximation for positive finite df.

    This preserves the source's polynomial approximation, not an exact Student t
    quantile. At small df it may be inaccurate or even have the wrong sign.
    A nonrepresentable result raises ArithmeticError. Median output is exactly zero.
    """
    degrees = finite(df, "df")
    if np.any(degrees <= 0):
        raise ValueError("df must be positive")
    z, degrees = np.broadcast_arrays(dinvnr(p, q), degrees)
    result, correction = z.copy(), np.zeros(z.shape)
    square = z * z
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for order, (coefficients, denominator) in enumerate(
            zip(_T_COEFFICIENTS, _T_DENOMINATORS, strict=True), 1
        ):
            term = (z * evaluate_polynomial(coefficients, square)) / denominator
            # Avoid forming df**order: it may underflow even when this term fits.
            for _ in range(order):
                term = term / degrees
            updated = result + term
            correction += np.where(
                np.abs(result) >= np.abs(term), (result - updated) + term, (term - updated) + result
            )
            result = updated
        result = result + correction
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("t starting approximation exceeds the finite output range")
    return _freeze(result)
