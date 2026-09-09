"""Legacy DCDFLIB normal contracts with unrestricted finite location and scale."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._cdflib import _freeze
from ._dcdflib import _probability_pair
from ._validation import FloatArray, finite
from .numerics import normal_tails


def _difference_ratio(a: FloatArray, b: FloatArray, divisor: FloatArray) -> FloatArray:
    with np.errstate(over="ignore"):
        difference = a - b
        result = difference / divisor
        overflow = ~np.isfinite(difference)
        # Overflowing subtraction implies opposite signs, so division of
        # each term cannot introduce cancellation between infinities.
        aa, bb = np.where(overflow, a, 0.0), np.where(overflow, b, 0.0)
        result = np.where(overflow, aa / divisor - bb / divisor, result)
    return result


def _affine(a: FloatArray, b: FloatArray, z: FloatArray) -> FloatArray:
    with np.errstate(over="ignore"):
        result = a + b * z
        failed = ~np.isfinite(result)
        # A product can overflow before a representable cancellation.
        result = np.where(failed, 2 * (a / 2 + (b / 2) * z), result)
    if np.any(~np.isfinite(result)):
        raise ValueError("computed normal location is not representable as a finite float")
    return result


def _tails(z: FloatArray) -> tuple[FloatArray, FloatArray]:
    p, q = normal_tails(z)
    recover = (np.minimum(p, q) == 0) & np.isfinite(z)
    if np.any(recover):
        small = np.exp(normal_tails(-np.abs(z[recover]), log=True)[0])
        p[recover] = np.where(z[recover] <= 0, small, 1.0)
        q[recover] = np.where(z[recover] <= 0, 1.0, small)
    return p, q


@dataclass(frozen=True)
class DCDFLIBNormal:
    """Computed group and owned immutable p/q, x, mean and sd arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    x: FloatArray
    mean: FloatArray
    sd: FloatArray


def cdfnor(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    x: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    sd: ArrayLike | None = None,
) -> DCDFLIBNormal:
    """Compute 1=p/q, 2=x, 3=mean or 4=sd for the legacy C/F77 interface.

    Omit the computed group. Input mean/sd default to 0/1. Finite locations
    and positive finite scales have no artificial magnitude bounds, including
    computed answers. Both inverse tails must be positive; either may be omitted.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    if any(v is not None for v in {1: (p, q), 2: (x,), 3: (mean,), 4: (sd,)}[which]):
        raise ValueError("omit the parameter group being computed")
    if which != 2 and x is None:
        raise ValueError("x is required")
    xx = finite(0.0 if x is None else x, "x")
    mu = finite(0.0 if mean is None else mean, "mean")
    sigma = finite(1.0 if sd is None else sd, "sd")
    if np.any(sigma <= 0):
        raise ValueError("sd must be positive")
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, xx, mu, sigma = np.broadcast_arrays(pp, qq, xx, mu, sigma)
    if which == 1:
        pp, qq = _tails(_difference_ratio(xx, mu, sigma))
    else:
        if np.any((pp <= 0) | (qq <= 0)):
            raise ValueError("normal inversion requires positive p and q")
        z = ndtri(np.minimum(pp, qq))
        z = np.where(pp <= qq, z, -z)
        if np.any(~np.isfinite(z)):
            raise ArithmeticError("normal quantile kernel failed")
        if which == 2:
            xx = _affine(mu, sigma, z)
        elif which == 3:
            mu = _affine(xx, -sigma, z)
        else:
            if np.any(z == 0):
                raise ValueError("median probability does not identify a unique sd")
            sigma = _difference_ratio(xx, mu, z)
            if np.any(~np.isfinite(sigma) | (sigma <= 0)):
                raise ValueError("computed sd must be positive and finite")
        lp, uq = _tails(_difference_ratio(xx, mu, sigma))
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("normal inversion failed forward verification")
    return DCDFLIBNormal(int(which), *map(_freeze, (pp, qq, xx, mu, sigma)))


def cumnor(x: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired standard-normal tails for the legacy cumulative interface."""
    result = cdfnor(x=x)
    return result.p, result.q
