"""Central/noncentral chi-square and F sampling with RANDLIB compatibility."""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, f, ncf, ncx2

from ._randlib_distributions import DistributionStream
from ._randlib_gamma import GammaSampler
from ._randlib_normal import standard_normal
from ._randlib_sampling import raw_batch
from ._validation import scalar
from .ranlist_random import _M1


def _gamma(shape: np.float32, source: str) -> GammaSampler:
    if shape <= 0:
        raise ValueError("legacy degrees of freedom produce a zero gamma shape")
    return GammaSampler(shape, source)


def _legacy(
    stream: DistributionStream,
    size: int,
    dfn: np.float32,
    dfd: np.float32 | None,
    nc: np.float32 | None,
) -> NDArray[np.float64]:
    k = np.float64 if stream.source == "c" else np.float32
    near_one = nc is not None and dfn < k(1.000001)
    numerator = (
        None
        if near_one
        else _gamma(np.float32((dfn - k(1) if nc is not None else dfn) / k(2)), stream.source)
    )
    denominator = None if dfd is None else _gamma(np.float32(dfd / k(2)), stream.source)
    shift = k(0) if nc is None else np.sqrt(k(nc))
    result = np.empty(size, dtype=np.float64)
    truncated = False
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for i in range(size):
            # The reference gfortran build evaluates the normal term first;
            # the reference C build evaluates the gamma term first.
            normal = (
                standard_normal(stream) + shift
                if nc is not None and stream.source == "fortran"
                else None
            )
            num = k(0) if numerator is None else k(2) * numerator.sample(stream)
            if nc is not None:
                if normal is None:
                    normal = standard_normal(stream) + shift
                num = num + normal * normal
            if dfd is None:
                result[i] = np.float32(num)
                continue
            # GENNF ignores the near-one numerator divisor, just as GENNCH
            # ignores its residual gamma contribution.
            xnum = np.float32(num if near_one else num / dfn)
            assert denominator is not None
            xden = np.float32(k(2) * denominator.sample(stream) / dfd)
            if xden <= k(1e-37) * xnum:
                result[i] = np.float32(1e37)
                truncated = True
            else:
                result[i] = xnum / xden
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite legacy chi-square/F samples; state unchanged")
    if truncated:
        warnings.warn("legacy F samples truncated to 1e37", RuntimeWarning, stacklevel=4)
    return result


def sample_chi_f(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    dfn: float,
    dfd: float | None,
    nc: float | None,
    legacy: bool,
    source: str,
    budget: int,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    dfn = scalar(dfn, "df" if dfd is None else "dfn")
    dfd = None if dfd is None else scalar(dfd, "dfd")
    nc = None if nc is None else scalar(nc, "noncentrality")
    if dfn <= 0 or dfd is not None and dfd <= 0:
        raise ValueError("degrees of freedom must be positive")
    if nc is not None and nc < 0:
        raise ValueError("noncentrality must be nonnegative")
    if legacy:
        if nc is not None and dfn < 1:
            raise ValueError("legacy noncentral numerator degrees of freedom must be >= 1")
        with np.errstate(over="ignore", under="ignore"):
            values = [np.float32(x) for x in (dfn, dfd, nc) if x is not None]
        originals = [x for x in (dfn, dfd, nc) if x is not None]
        if any(not np.isfinite(x) or x == 0 and y != 0 for x, y in zip(values, originals)):
            raise ValueError("legacy parameters must be representable in float32")
        stream = DistributionStream(state, antithetic, source, budget)
        result = _legacy(
            stream,
            size,
            np.float32(dfn),
            None if dfd is None else np.float32(dfd),
            None if nc is None else np.float32(nc),
        )
        state = stream.state
    else:
        if size > budget:
            raise ArithmeticError("distribution sampling exceeded max_attempts; state unchanged")
        raw, state = raw_batch(state, size, antithetic)
        uniform = raw / _M1
        parameters: tuple[float, ...]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            if dfd is None:
                distribution, parameters = (
                    (chi2, (dfn,)) if nc is None or nc == 0 else (ncx2, (dfn, nc))
                )
            else:
                distribution, parameters = (
                    (f, (dfn, dfd)) if nc is None or nc == 0 else (ncf, (dfn, dfd, nc))
                )
            result = distribution.ppf(uniform, *parameters)
            # Some extreme F quantiles saturate at finite library limits.
            # Validate the smaller tail rather than accepting a finite sentinel.
            lower = uniform <= 0.5
            probability = np.empty(size)
            probability[lower] = distribution.cdf(result[lower], *parameters)
            probability[~lower] = distribution.sf(result[~lower], *parameters)
            target = np.minimum(uniform, 1 - uniform)
            inaccurate = (result > 0) & (
                ~np.isfinite(probability)
                | (np.abs(probability - target) > 64 * np.finfo(float).eps + 1e-7 * target)
            )
        if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(inaccurate):
            raise ArithmeticError("nonfinite or invalid chi-square/F samples; state unchanged")
    result.flags.writeable = False
    return result, state
