"""Discrete one-sided binomial critical regions and achieved power."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

    lo = _maximum_region(n, p0, lower_tail, level)
    critical, significance = _binomial_region(n, lower_tail, lo, p0)
    _, power = _binomial_region(n, lower_tail, lo, pa)
    next_critical, next_significance = _binomial_region(n, lower_tail, lo + 1, p0)
    _, next_power = _binomial_region(n, lower_tail, lo + 1, pa)
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


def _binomial_region(
    n: FloatArray, lower_tail: NDArray[np.bool_], included: FloatArray, p: FloatArray
) -> tuple[FloatArray, FloatArray]:
    critical = np.where(lower_tail, included - 1, n - included + 1)
    result = binomial_test(np.clip(critical, 0, n), n, p)
    tail = np.where(lower_tail, result.p_less, result.p_greater)
    tail = np.where(included == 0, 0, np.where(included == n + 1, 1, tail))
    return critical, tail


@dataclass(frozen=True)
class BinomialSignificance:
    """Least permissive test reaching target_power, with its previous candidate.

    previous_* excludes one more event count and has power below target_power.
    A full rejection region has significance=power=1; it is reported explicitly.
    The direction is determined by alternative_probability versus null_probability.
    """

    trials: FloatArray
    null_probability: FloatArray
    alternative_probability: FloatArray
    target_power: FloatArray
    critical: FloatArray
    significance: FloatArray
    power: FloatArray
    previous_critical: FloatArray
    previous_significance: FloatArray
    previous_power: FloatArray


def binomial_significance(
    trials: ArrayLike,
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    target_power: ArrayLike = 0.8,
) -> BinomialSignificance:
    """Find the minimum achieved significance that attains target_power.

    Inputs broadcast. The direction follows the alternative; probabilities and
    target_power must lie strictly in (0,1), and null and alternative must differ.
    Uses discrete count bisection, including empty/full candidate regions.
    """
    n, p0, pa, target = np.broadcast_arrays(
        count(trials, "trials"),
        finite(null_probability, "null_probability"),
        finite(alternative_probability, "alternative_probability"),
        finite(target_power, "target_power"),
    )
    if np.any(
        (n < 1)
        | (p0 <= 0)
        | (p0 >= 1)
        | (pa <= 0)
        | (pa >= 1)
        | (p0 == pa)
        | (target <= 0)
        | (target >= 1)
    ):
        raise ValueError(
            "Require positive integral trials, distinct interior probabilities and 0<target_power<1"
        )
    lower_tail = pa < p0
    hi = _minimum_region(n, pa, lower_tail, target)
    critical, significance = _binomial_region(n, lower_tail, hi, p0)
    _, power = _binomial_region(n, lower_tail, hi, pa)
    previous_critical, previous_significance = _binomial_region(n, lower_tail, hi - 1, p0)
    _, previous_power = _binomial_region(n, lower_tail, hi - 1, pa)
    return BinomialSignificance(
        n,
        p0,
        pa,
        target,
        critical,
        significance,
        power,
        previous_critical,
        previous_significance,
        previous_power,
    )


def _maximum_region(
    n: FloatArray, p0: FloatArray, lower_tail: NDArray[np.bool_], level: FloatArray
) -> FloatArray:
    lo, hi = np.zeros(n.shape), n + 1
    while np.any(hi - lo > 1):
        mid = lo + np.floor((hi - lo) / 2)
        _, size = _binomial_region(n, lower_tail, mid, p0)
        acceptable = size <= level
        lo, hi = np.where(acceptable, mid, lo), np.where(acceptable, hi, mid)
    return lo


def _minimum_region(
    n: FloatArray, pa: FloatArray, lower_tail: NDArray[np.bool_], target: FloatArray
) -> FloatArray:
    lo, hi = np.zeros(n.shape), n + 1
    while np.any(hi - lo > 1):
        mid = lo + np.floor((hi - lo) / 2)
        _, power = _binomial_region(n, lower_tail, mid, pa)
        adequate = power >= target
        lo, hi = np.where(adequate, lo, mid), np.where(adequate, mid, hi)
    return hi


def _probability_bracket(
    n: FloatArray,
    lower: NDArray[np.bool_],
    included: FloatArray,
    target: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    guess: FloatArray,
    equality_right: bool,
) -> tuple[FloatArray, FloatArray]:
    """Refine a monotone binomial-tail crossing, choosing which side owns equality."""
    if np.any(~np.isfinite(guess)):
        raise ArithmeticError("Inverse beta could not produce a finite probability")
    guess = np.clip(guess, lo, hi)
    _, value = _binomial_region(n, lower, included, guess)
    right = np.where(lower, value > target, value < target) | ((value == target) & equality_right)
    lo, hi = np.where(right, guess, lo), np.where(right, hi, guess)
    for _ in range(1076):
        mid = lo + (hi - lo) / 2
        active = (mid > lo) & (mid < hi)
        if not np.any(active):
            return lo, hi
        _, value = _binomial_region(n, lower, included, mid)
        right = np.where(lower, value > target, value < target) | (
            (value == target) & equality_right
        )
        lo, hi = np.where(active & right, mid, lo), np.where(active & ~right, mid, hi)
    raise ArithmeticError("Binomial probability bracket failed to converge")
