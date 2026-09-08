"""CDFLIB90 normal tails and closed-form parameter inversions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .numerics import normal_tails


def _bounded(value: ArrayLike | None, name: str, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = finite(value, name)
    lo = 1e-10 if name == "sd" else -1e100
    if computed:
        tolerance = 8 * np.finfo(float).eps
        array = np.where((array < lo) & (lo - array <= tolerance * abs(lo)), lo, array)
        array = np.where((array > 1e100) & (array - 1e100 <= tolerance * 1e100), 1e100, array)
    if np.any((array < lo) | (array > 1e100)):
        raise ValueError(f"{name} must lie in [{lo:g},1e100]")
    return array


@dataclass(frozen=True)
class CDFNormal:
    """All five broadcast arrays and the computed group, owned and immutable."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    x: FloatArray
    mean: FloatArray
    sd: FloatArray


def cdf_normal(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    x: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    sd: ArrayLike | None = None,
) -> CDFNormal:
    """Compute 1=tails, 2=x, 3=mean, or 4=sd; omit the computed group.

    Input mean/sd default to 0/1. Locations lie in [-1e100,1e100], SD in
    [1e-10,1e100], including computed answers. Inversion accepts any positive
    representable complementary probabilities, extending the source's 1e-10
    tail cutoff. Endpoints, unidentifiable median SD and invalid solutions fail.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3, 4)
    ):
        raise ValueError("which must be 1, 2, 3 or 4")
    outputs = {1: (cum, ccum), 2: (x,), 3: (mean,), 4: (sd,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    xx = _bounded(x, "x") if which != 2 else np.asarray(0.0)
    mu = _bounded(0.0 if mean is None else mean, "mean")
    sigma = _bounded(1.0 if sd is None else sd, "sd")
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, xx, mu, sigma = np.broadcast_arrays(p, q, xx, mu, sigma)
    if which == 1:
        p, q = normal_tails((xx - mu) / sigma)
    else:
        if np.any((p <= 0) | (q <= 0)):
            raise ValueError("normal parameter inversion requires positive cum and ccum")
        z = ndtri(np.minimum(p, q))
        z = np.where(p <= q, z, -z)
        if np.any(~np.isfinite(z)):
            raise ArithmeticError("normal quantile kernel failed")
        if which == 2:
            xx = _bounded(mu + sigma * z, "x", computed=True)
        elif which == 3:
            mu = _bounded(xx - sigma * z, "mean", computed=True)
        else:
            if np.any(z == 0):
                raise ValueError("median probability does not identify a unique positive sd")
            sigma = _bounded((xx - mu) / z, "sd", computed=True)
    return CDFNormal(int(which), *map(_freeze, (p, q, xx, mu, sigma)))


def cum_normal(x: ArrayLike, mean: ArrayLike = 0, sd: ArrayLike = 1) -> FloatArray:
    """Normal lower tail with optional location and standard deviation."""
    return cdf_normal(x=x, mean=mean, sd=sd).cum


def ccum_normal(x: ArrayLike, mean: ArrayLike = 0, sd: ArrayLike = 1) -> FloatArray:
    """Normal upper tail evaluated directly, without subtracting a rounded CDF."""
    return cdf_normal(x=x, mean=mean, sd=sd).ccum


def inv_normal(
    cum: ArrayLike | None = None,
    mean: ArrayLike = 0,
    sd: ArrayLike = 1,
    *,
    ccum: ArrayLike | None = None,
) -> FloatArray:
    """Normal quantile using the smaller supplied probability tail."""
    return cdf_normal(2, cum=cum, ccum=ccum, mean=mean, sd=sd).x
