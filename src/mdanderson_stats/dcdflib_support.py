"""Legacy DCDFLIB machine model and checked, vectorized translation helpers."""

from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_beta_asymptotic import basym
from .cdflib_beta_factors import brcmp1, brcomp
from .cdflib_beta_fraction import bfrac
from .cdflib_beta_ratio import bratio
from .cdflib_beta_series import apser, bgrat, bpser, fpser
from .cdflib_beta_shift import bup
from .cdflib_beta_support import betaln
from .cdflib_elementary import alnrel, evaluate_polynomial, rexp, rlog, rlog1
from .cdflib_error_exponential import _integer, erfc1, esum
from .cdflib_error_exponential import erf as erf1
from .cdflib_gamma_factor import rcomp
from .cdflib_gamma_ratios import algdiv, bcorr, gsumln
from .cdflib_gamma_support import alngam, gam1, gamln, gamln1, psi
from .cdflib_gamma_support import gamma as Xgamm
from .cdflib_incomplete_gamma import grat1, gratio

# Archived int32 / IEEE binary32 / IEEE binary64 model, independent of C long.
_IMACH = (2, 31, 2147483647, 2, 24, -125, 128, 53, -1021, 1024)
_FLOAT = np.finfo(np.float64)
_SPMACH = (float(_FLOAT.eps), float(_FLOAT.tiny), float(_FLOAT.max))


def _index(i: int, limit: int) -> int:
    if isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= limit:
        raise ValueError(f"index must be an integer in 1..{limit}")
    return i - 1


def ipmpar(i: int) -> int:
    """Return one of the ten archived integer/machine-model parameters."""
    return _IMACH[_index(i, 10)]


def spmpar(i: int) -> float:
    """Return binary64 epsilon, smallest normal, or largest finite value (i=1..3)."""
    return _SPMACH[_index(i, 3)]


def exparg(l: ArrayLike) -> FloatArray:  # noqa: E741 - native public argument name
    """Legacy conservative exponential limits, distinct from the F95 exparg.

    Retain the source's 0.99999 margin and rounded log(radix). A nonzero signed
    int32 selector chooses the lower normal-range limit; zero chooses upper.
    """
    flag = _integer(l, "l")
    exponent = np.where(flag == 0, ipmpar(10), ipmpar(9) - 1)
    return _freeze(0.99999 * (exponent * 0.69314718055995))


def devlpl(a: ArrayLike, n: int, x: ArrayLike) -> FloatArray:
    """Evaluate the first n ascending-power coefficients at a batch of x values."""
    coefficients = finite(a, "a")
    if coefficients.ndim != 1:
        raise ValueError("a must be one-dimensional")
    _index(n, coefficients.size)
    return evaluate_polynomial(coefficients[:n], x)


def fifdint(a: ArrayLike) -> FloatArray:
    """Truncate finite values toward zero, returning binary64 with native positive zero."""
    value = np.trunc(finite(a, "a"))
    return _freeze(np.where(value == 0, 0.0, value))


def fifdmax1(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Elementwise maximum; equal values select a, preserving its zero sign."""
    left, right = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    return _freeze(np.where(left < right, right, left))


def fifdmin1(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Elementwise minimum; equal values select b, preserving its zero sign."""
    left, right = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    return _freeze(np.where(left < right, left, right))


def fifdsign(mag: ArrayLike, sign: ArrayLike) -> FloatArray:
    """Apply the native comparison-based sign transfer, including signed-zero quirks."""
    value, source = np.broadcast_arrays(finite(mag, "mag"), finite(sign, "sign"))
    magnitude = np.where(value < 0, -value, value)
    return _freeze(np.where(source < 0, -magnitude, magnitude))


def _freeze_int(value: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(value.tobytes(), dtype=np.int64).reshape(value.shape)


def fifidint(a: ArrayLike) -> NDArray[np.int64]:
    """Truncate to checked int64, replacing platform-dependent C long casts."""
    value = finite(a, "a")
    if np.any((value < -(2.0**63)) | (value >= 2.0**63)):
        raise ValueError("Truncated values must fit int64")
    return _freeze_int(value.astype(np.int64))


def _integers(a: ArrayLike, name: str) -> NDArray[np.int64]:
    value = np.asarray(a)
    if value.dtype.kind not in "iu" or (
        value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must contain integers in the int64 range")
    return value.astype(np.int64)


def fifmod(a: ArrayLike, b: ArrayLike) -> NDArray[np.int64]:
    """Exact int64 remainder with the dividend's sign, as in C/Fortran MOD."""
    left, right = np.broadcast_arrays(_integers(a, "a"), _integers(b, "b"))
    if np.any(right == 0):
        raise ValueError("Divisors must be nonzero")
    # min_int / -1 overflows a C quotient; its mathematical remainder is zero.
    safe = np.where(right == -1, 1, right)
    result = np.asarray(np.remainder(left, safe))
    adjust = (result != 0) & ((left < 0) != (safe < 0))
    np.subtract(result, safe, out=result, where=adjust)
    return _freeze_int(result)


def ftnstop(message: str | None = None) -> NoReturn:
    """Raise RuntimeError instead of writing stderr and terminating the process."""
    if message is not None and not isinstance(message, str):
        raise ValueError("message must be a string or None")
    raise RuntimeError("Fortran STOP" if message is None else message)


__all__ = [
    "algdiv",
    "alngam",
    "alnrel",
    "apser",
    "basym",
    "bcorr",
    "betaln",
    "bfrac",
    "bgrat",
    "bpser",
    "bratio",
    "brcmp1",
    "brcomp",
    "bup",
    "erf1",
    "erfc1",
    "esum",
    "exparg",
    "fpser",
    "gam1",
    "gamln",
    "gamln1",
    "Xgamm",
    "grat1",
    "gratio",
    "gsumln",
    "psi",
    "rcomp",
    "rexp",
    "rlog",
    "rlog1",
    "ipmpar",
    "spmpar",
    "devlpl",
    "fifdint",
    "fifdmax1",
    "fifdmin1",
    "fifdsign",
    "fifidint",
    "fifmod",
    "ftnstop",
]
