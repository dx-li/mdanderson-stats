"""Legacy DCDFLIB Poisson contracts, including zero mean and real counts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._dcdflib import _invert_positive, _probability_pair
from ._validation import FloatArray, finite
from .dcdflib_gamma import _scaled_quantile
from .dcdflib_gamma import _tails as _gamma_tails


def _nonnegative(value: ArrayLike | None, name: str) -> FloatArray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = finite(value, name)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _tails(s: FloatArray, mean: FloatArray) -> tuple[FloatArray, FloatArray]:
    # Reduce directly to gamma, avoiding the native doubling of mean and s+1.
    q, p = _gamma_tails(mean, s + 1, np.ones(s.shape))
    zero = s == 0
    p[zero], q[zero] = np.exp(-mean[zero]), -np.expm1(-mean[zero])
    return p, q


@dataclass(frozen=True)
class DCDFLIBPoisson:
    """Computed group and owned immutable p/q, count and mean arrays."""

    which: int
    p: FloatArray
    q: FloatArray
    s: FloatArray
    mean: FloatArray


def cdfpoi(
    which: int = 1,
    *,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    s: ArrayLike | None = None,
    mean: ArrayLike | None = None,
) -> DCDFLIBPoisson:
    """Compute 1=p/q, 2=real count s or 3=mean; omit the computed group.

    Finite nonnegative inputs have no upper cap; computed s/mean lie in
    [0,1e100]. This is the continuous gamma extension, not an integer PPF.
    Source XLAM is named mean. Zero mean has p=1 and q=0 at every count.
    """
    if (
        isinstance(which, (bool, np.bool_))
        or not isinstance(which, (int, np.integer))
        or which not in (1, 2, 3)
    ):
        raise ValueError("which must be 1, 2 or 3")
    if any(v is not None for v in {1: (p, q), 2: (s,), 3: (mean,)}[which]):
        raise ValueError("omit the parameter group being computed")
    count = _nonnegative(s, "s") if which != 2 else np.asarray(0.0)
    mu = _nonnegative(mean, "mean") if which != 3 else np.asarray(1.0)
    pp, qq = _probability_pair(p, q) if which != 1 else (np.asarray(0.5), np.asarray(0.5))
    pp, qq, count, mu = np.broadcast_arrays(pp, qq, count, mu)
    if which == 1:
        pp, qq = _tails(count, mu)
    else:
        if np.any((pp <= 0) | (qq <= 0)):
            raise ValueError("legacy Poisson inversion requires positive p and q")
        if which == 2:
            if np.any(mu <= 0):
                raise ValueError("count inversion requires positive mean")
            fixed, lower = mu.ravel(), (pp <= qq).ravel()

            def evaluate(shape: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
                lp, uq = _tails(shape - 1, fixed[indices])
                return np.where(lower[indices], lp, uq)

            shape = _invert_positive(
                np.minimum(pp, qq),
                np.ones(pp.shape),
                np.full(pp.shape, 1e100),
                evaluate,
                initial=np.where(mu >= 1, mu, 5.0),
                unbracketed_message="Poisson count solution lies outside [0,1e100]",
            )
            count = shape - 1
        else:
            mu = _scaled_quantile(qq, pp, count + 1, np.ones(pp.shape))
            mu = np.where((mu > 1e100) & (mu <= 1e100 * (1 + 8 * np.finfo(float).eps)), 1e100, mu)
            if np.any(~np.isfinite(mu) | (mu <= 0) | (mu > 1e100)):
                raise ValueError("Poisson mean is not representable within [0,1e100]")
        lp, uq = _tails(count, mu)
        target = np.minimum(pp, qq)
        if np.any(
            np.abs(np.where(pp <= qq, lp, uq) - target)
            > 1e-7 * target + 32 * np.nextafter(0.0, 1.0)
        ):
            raise ArithmeticError("legacy Poisson inversion failed forward verification")
    return DCDFLIBPoisson(int(which), *map(_freeze, (pp, qq, count, mu)))


def cumpoi(s: ArrayLike, mean: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Paired Poisson tails, continuously extended to nonnegative real counts."""
    result = cdfpoi(s=s, mean=mean)
    return result.p, result.q
