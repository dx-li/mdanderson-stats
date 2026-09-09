"""Vectorized error functions and exponential helpers from CDFLIB F95."""

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from ._cdflib import _freeze
from ._validation import FloatArray, finite


def _integer(value: ArrayLike, name: str) -> FloatArray:
    result = finite(value, name)
    if np.any((result != np.floor(result)) | (result < -(2**31)) | (result >= 2**31)):
        raise ValueError(f"{name} must contain signed 32-bit integers")
    return result


def erf(x: ArrayLike) -> FloatArray:
    """Evaluate the real error function for finite scalar or array coordinates."""
    return _freeze(special.erf(finite(x, "x")))


def erfc1(ind: ArrayLike, x: ArrayLike) -> FloatArray:
    """Evaluate erfc(x) for ind=0, or exp(x*x)*erfc(x) for nonzero integer ind.

    Inputs broadcast. Positive subnormal tails are retained; a scaled result
    exceeding finite float64 raises ArithmeticError.
    """
    flag, value = np.broadcast_arrays(_integer(ind, "ind"), finite(x, "x"))
    scaled = flag != 0
    result = np.empty(value.shape)
    with np.errstate(over="ignore", under="ignore"):
        result[scaled] = special.erfcx(value[scaled])
        regular = ~scaled & (value <= 26)
        result[regular] = special.erfc(value[regular])
        tail = ~scaled & (value > 26)
        z = value[tail]
        # erfc's compiled kernel may cut off subnormal results. The scaled
        # function and split exponential keep intermediates normal until the
        # final multiplication, avoiding a second subnormal rounding.
        half_exponential = np.exp((-0.5 * z) * z)
        result[tail] = (half_exponential * special.erfcx(z)) * half_exponential
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("scaled complementary error function exceeds the finite output range")
    return _freeze(result)


def esum(mu: ArrayLike, x: ArrayLike) -> FloatArray:
    """Compute exp(mu+x) with a signed int32 mu and finite x, broadcasting inputs.

    Combine the exponent before exponentiation to avoid intermediate overflow
    and underflow. A nonfinite result raises ArithmeticError.
    """
    integer, value = np.broadcast_arrays(_integer(mu, "mu"), finite(x, "x"))
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(integer + value)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("exponential sum exceeds the finite output range")
    return _freeze(result)


def exparg(l: ArrayLike) -> FloatArray:  # noqa: E741 - native public argument name
    """Return log(max float64) for l=0, otherwise log(min NORMAL float64).

    This preserves the native executable contract. Subnormal nonzero
    exponentials remain possible below the returned negative threshold.
    """
    flag = _integer(l, "l")
    limits = np.finfo(np.float64)
    return _freeze(np.where(flag == 0, np.log(limits.max), np.log(limits.tiny)))
