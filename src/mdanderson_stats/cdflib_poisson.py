"""CDFLIB90 Poisson tails and inversions with continuous real counts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_gamma import _invert_shape, _tails, cdf_gamma


def _parameter(value: ArrayLike | None, name: str, *, computed: bool = False) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    low = 0.0 if name == "s" else 1e-10
    if computed:
        tolerance = 8 * np.finfo(float).eps
        result = np.where((result < low) & (low - result <= tolerance * low), low, result)
        result = np.where((result > 1e100) & (result - 1e100 <= tolerance * 1e100), 1e100, result)
    if np.any((result < low) | (result > 1e100)):
        raise ValueError(f"{name} must lie in [{low:g},1e100]")
    return result


@dataclass(frozen=True)
class CDFPoisson:
    """Computed group and immutable broadcast probability, count and mean arrays."""

    which: int
    cum: FloatArray
    ccum: FloatArray
    s: FloatArray
    mean: FloatArray


def cdf_poisson(
    which: int = 1,
    *,
    cum: ArrayLike | None = None,
    ccum: ArrayLike | None = None,
    s: ArrayLike | None = None,
    mean: ArrayLike | None = None,
) -> CDFPoisson:
    """Compute 1=tails, 2=real count s or 3=mean; omit the computed group.

    This is CDFLIB90's continuous gamma extension, not an integer-valued PPF.
    Counts lie in [0,1e100]; mean (the archived LAMBDA) in [1e-10,1e100].
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    outputs = {1: (cum, ccum), 2: (s,), 3: (mean,)}
    if any(value is not None for value in outputs[which]):
        raise ValueError("omit the parameter group being computed")
    count = _parameter(s, "s") if which != 2 else np.asarray(0.0)
    mu = _parameter(mean, "mean") if which != 3 else np.asarray(1.0)
    p, q = _pair(cum, ccum, "cum/ccum") if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    p, q, count, mu = np.broadcast_arrays(p, q, count, mu)
    if which == 1:
        q, p = _tails(count + 1, mu)
    elif which == 2:
        if np.any((p <= 0) | (q <= 0)):
            raise ValueError("count inversion requires positive cum and ccum")
        # Keep the s=0 endpoint despite a few units of probability roundoff.
        lp, uq = _tails(np.ones(p.shape), mu)
        lower, target = q <= p, np.minimum(p, q)
        endpoint = np.where(lower, lp, uq)
        close = np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint
        target = np.where(close, endpoint, target)
        gp, gq = np.where(lower, target, 1 - target), np.where(lower, 1 - target, target)
        count = _parameter(_invert_shape(gp, gq, mu, 1.0, 1e100) - 1, "s", computed=True)
    else:
        gamma = cdf_gamma(2, cum=q, ccum=p, shape=count + 1)
        mu = _parameter(gamma.x, "mean", computed=True)
    return CDFPoisson(int(which), *map(_freeze, (p, q, count, mu)))


def cum_poisson(s: ArrayLike, mean: ArrayLike) -> FloatArray:
    """Lower Poisson tail extended to nonnegative real counts."""
    return cdf_poisson(s=s, mean=mean).cum


def ccum_poisson(s: ArrayLike, mean: ArrayLike) -> FloatArray:
    """Direct upper Poisson tail extended to nonnegative real counts."""
    return cdf_poisson(s=s, mean=mean).ccum


def inv_poisson(
    cum: ArrayLike | None, mean: ArrayLike, *, ccum: ArrayLike | None = None
) -> FloatArray:
    """Continuous count inverse; no integer rounding is performed."""
    return cdf_poisson(2, cum=cum, ccum=ccum, mean=mean).s
