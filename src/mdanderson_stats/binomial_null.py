"""Null-probability inversion for a one-sided binomial design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainccinv, betaincinv

from ._validation import FloatArray, count, finite
from .binomial_design import _binomial_region, _minimum_region, _probability_bracket


@dataclass(frozen=True)
class BinomialNull:
    trials: FloatArray
    alternative_probability: FloatArray
    alpha: FloatArray
    target_power: FloatArray
    alternative: str
    null_probability: FloatArray
    critical: FloatArray
    significance: FloatArray
    power: FloatArray
    previous_power: FloatArray
    probability_lower: FloatArray
    probability_upper: FloatArray


def binomial_null(
    trials: ArrayLike,
    alternative_probability: ArrayLike,
    alpha: ArrayLike = 0.05,
    target_power: ArrayLike = 0.8,
    *,
    alternative: str = "greater",
) -> BinomialNull:
    """Closest null admitting a size-controlled region with the requested power.

    The least permissive region attaining target_power at the supplied alternative
    is selected first. Its null probability is inverted at alpha. A full region
    cannot have significance below one and raises. Inputs broadcast; require
    positive integral trials, an interior alternative and 0<alpha<=target_power<1.
    Returned brackets are adjacent binary64 probabilities for the evaluated tail.
    """
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be less or greater")
    n, pa, level, target = np.broadcast_arrays(
        count(trials, "trials"),
        finite(alternative_probability, "alternative_probability"),
        finite(alpha, "alpha"),
        finite(target_power, "target_power"),
    )
    if np.any(
        (n < 1)
        | (pa <= 0)
        | (pa >= 1)
        | (level <= 0)
        | (level >= 1)
        | (target < level)
        | (target >= 1)
    ):
        raise ValueError("Require positive integral trials, 0<pa<1 and 0<alpha<=target_power<1")
    lower = np.full(n.shape, alternative == "less")
    included = _minimum_region(n, pa, lower, target)
    if np.any(included == n + 1):
        raise ValueError("Required full rejection region cannot satisfy alpha below one")
    critical, power = _binomial_region(n, lower, included, pa)
    _, previous_power = _binomial_region(n, lower, included - 1, pa)
    guess = (
        betainccinv(critical + 1, n - critical, level)
        if alternative == "less"
        else betaincinv(critical, n - critical + 1, level)
    )
    lo, hi = _probability_bracket(
        n,
        lower,
        included,
        level,
        np.where(lower, pa, 0),
        np.where(lower, 1, pa),
        guess,
        equality_right=alternative == "greater",
    )
    p0 = np.where(lower, hi, lo)
    p0 = np.where(power == level, pa, p0)
    lo = np.where(power == level, pa, lo)
    hi = np.where(power == level, pa, hi)
    _, size = _binomial_region(n, lower, included, p0)
    if np.any(size > level) or np.any(power < target):
        raise ArithmeticError("Returned null does not satisfy significance and power constraints")
    return BinomialNull(
        n, pa, level, target, alternative, p0, critical, size, power, previous_power, lo, hi
    )
