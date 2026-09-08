"""Discrete one-sided binomial critical regions and achieved power."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .onesample import binomial_test


@dataclass(frozen=True)
class BinomialPower:
    """Selected test and adjacent more permissive test, all broadcast arrays.

    Reject at or below critical when alternative_probability < null_probability,
    otherwise at or above critical. An empty region has critical=-1 (less) or
    trials+1 (greater), with zero significance/power. next_* describes inclusion
    of one additional event count and therefore exceeds the requested alpha.
    """

    trials: FloatArray
    null_probability: FloatArray
    alternative_probability: FloatArray
    alpha: FloatArray
    critical: FloatArray
    significance: FloatArray
    power: FloatArray
    next_critical: FloatArray
    next_significance: FloatArray
    next_power: FloatArray


def binomial_power(
    trials: ArrayLike,
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    alpha: ArrayLike = 0.05,
) -> BinomialPower:
    """Choose the largest nonrandomized one-sided region with size <= alpha.

    All inputs broadcast. Counts are integral; probabilities are strictly inside
    (0,1), null and alternative differ, and 0<alpha<1. Integer bisection selects
    the cutoff without continuous quantile rounding. No normal approximation is
    used. Empty critical regions are valid explicit results, not solver failures.
    """
    n, p0, pa, level = np.broadcast_arrays(
        count(trials, "trials"),
        finite(null_probability, "null_probability"),
        finite(alternative_probability, "alternative_probability"),
        finite(alpha, "alpha"),
    )
    if np.any(
        (n < 1)
        | (p0 <= 0)
        | (p0 >= 1)
        | (pa <= 0)
        | (pa >= 1)
        | (p0 == pa)
        | (level <= 0)
        | (level >= 1)
    ):
        raise ValueError(
            "Require positive integral trials, distinct interior probabilities and 0<alpha<1"
        )
    lower_tail = pa < p0

    def evaluate(included: FloatArray, p: FloatArray) -> tuple[FloatArray, FloatArray]:
        critical = np.where(lower_tail, included - 1, n - included + 1)
        result = binomial_test(np.clip(critical, 0, n), n, p)
        tail = np.where(lower_tail, result.p_less, result.p_greater)
        tail = np.where(included == 0, 0, np.where(included == n + 1, 1, tail))
        return critical, tail

    lo, hi = np.zeros(n.shape), n + 1
    while np.any(hi - lo > 1):
        mid = lo + np.floor((hi - lo) / 2)
        _, size = evaluate(mid, p0)
        acceptable = size <= level
        lo, hi = np.where(acceptable, mid, lo), np.where(acceptable, hi, mid)
    critical, significance = evaluate(lo, p0)
    _, power = evaluate(lo, pa)
    next_critical, next_significance = evaluate(lo + 1, p0)
    _, next_power = evaluate(lo + 1, pa)
    return BinomialPower(
        n,
        p0,
        pa,
        level,
        critical,
        significance,
        power,
        next_critical,
        next_significance,
        next_power,
    )
