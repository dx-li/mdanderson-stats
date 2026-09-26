"""Conditional exact power for two independent Poisson event counts."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import poisson

from ._validation import FloatArray, scalar
from .stplan_continuous import _alpha, _inputs, _power
from .stplan_discrete import _binomial_exact_power

_MAX_SUPPORT_TERMS = 200_000


def stplan_poisson_two_sample_power(
    rate1: ArrayLike,
    rate2: ArrayLike,
    exposure1: ArrayLike,
    exposure2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
    tail_tolerance: float = 1e-10,
) -> FloatArray:
    """Power for the conditional exact binomial comparison of two Poisson rates.

    Given total events ``N``, the first-arm count is binomial with null
    probability ``exposure1 / (exposure1 + exposure2)`` and alternative
    probability proportional to its expected events. Power is averaged over
    ``N ~ Poisson(rate1*exposure1 + rate2*exposure2)``. The sum includes zero
    events and is left unnormalized; its absolute omitted upper-tail mass is at
    most ``tail_tolerance``. This avoids STPLAN's large-event approximation,
    truncated/renormalized generator, and empty-test sentinel behavior.

    Total support work across all broadcast cases is limited to 200,000 terms.
    """
    a = _alpha(alpha, sides)
    tail = scalar(tail_tolerance, "tail_tolerance")
    if not 0 < tail < 0.01:
        raise ValueError("tail_tolerance must lie strictly between 0 and 0.01")
    r1, r2, t1, t2 = _inputs(
        (rate1, "rate1"),
        (rate2, "rate2"),
        (exposure1, "exposure1"),
        (exposure2, "exposure2"),
    )
    if np.any(r1 < 0) or np.any(r2 < 0) or np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("rates must be nonnegative and exposures positive")
    mean1, mean2 = r1 * t1, r2 * t2
    total_mean = mean1 + mean2
    if np.any(~np.isfinite(total_mean)):
        raise ArithmeticError("expected total event count is not representable")
    exposure_scale = np.maximum(t1, t2)
    p_null = (t1 / exposure_scale) / (t1 / exposure_scale + t2 / exposure_scale)
    safe_total = np.where(total_mean > 0, total_mean, 1.0)
    p_alt = np.where(total_mean > 0, mean1 / safe_total, p_null)

    upper = np.maximum(0.0, np.ceil(poisson.isf(tail, total_mean)))
    if np.any(~np.isfinite(upper) | (upper >= 2**53)):
        raise ValueError("Poisson upper-tail cutoff is not representable")
    while np.any(poisson.sf(upper, total_mean) > tail):
        upper = np.where(poisson.sf(upper, total_mean) > tail, upper + 1, upper)
        if np.any(upper >= 2**53):
            raise ArithmeticError("could not bound the omitted Poisson tail")
    if np.any(upper + 1 > _MAX_SUPPORT_TERMS):
        raise ValueError("Poisson mixture exceeds 200000 support terms")
    work = sum(int(value) + 1 for value in upper.flat)
    if work > _MAX_SUPPORT_TERMS:
        raise ValueError("Poisson mixture exceeds 200000 total support terms")
    if total_mean.size == 0:
        return np.empty(total_mean.shape, dtype=float)

    flat_upper = upper.ravel().astype(np.int64)
    lengths = flat_upper + 1
    case_index: NDArray[np.int64] = np.repeat(np.arange(flat_upper.size), lengths)
    starts: NDArray[np.int64] = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    events: NDArray[np.int64] = np.arange(work, dtype=np.int64) - starts
    flat_null = p_null.ravel()
    flat_alt = p_alt.ravel()
    flat_mean = total_mean.ravel()
    result = np.zeros(flat_upper.size, dtype=float)
    for start in range(0, work, 2048):
        stop = min(start + 2048, work)
        idx = case_index[start:stop]
        counts = events[start:stop]
        conditional = _binomial_exact_power(flat_null[idx], flat_alt[idx], counts.astype(float), a)
        weights = poisson.pmf(counts, flat_mean[idx])
        np.add.at(result, idx, weights * conditional)
    return _power(result.reshape(total_mean.shape))
