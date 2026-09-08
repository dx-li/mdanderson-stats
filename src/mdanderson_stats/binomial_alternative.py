"""Alternative probability inversion for a fixed one-sided binomial test."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainccinv, betaincinv

from ._validation import FloatArray, count, finite
from .binomial_design import _binomial_region, _maximum_region, _probability_bracket


@dataclass(frozen=True)
class BinomialAlternative:
    trials: FloatArray
    null_probability: FloatArray
    alpha: FloatArray
    target_power: FloatArray
    alternative: str
    alternative_probability: FloatArray
    critical: FloatArray
    significance: FloatArray
    power: FloatArray
    probability_lower: FloatArray
    probability_upper: FloatArray


def binomial_alternative(
    trials: ArrayLike,
    null_probability: ArrayLike,
    alpha: ArrayLike = 0.05,
    target_power: ArrayLike = 0.8,
    *,
    alternative: str = "greater",
) -> BinomialAlternative:
    """Closest alternative reaching target power for the fixed size-controlled test.

    Inputs broadcast. alternative is 'less' or 'greater'; alpha and target_power
    are interior probabilities with target_power>=alpha. Empty critical regions
    raise since they cannot achieve positive power for any alternative probability.
    A beta inverse seeds a bracket refined to adjacent binary64 probabilities;
    the returned endpoint attains the target under the evaluated binomial tail.
    """
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be less or greater")
    n, p0, level, target = np.broadcast_arrays(
        count(trials, "trials"),
        finite(null_probability, "null_probability"),
        finite(alpha, "alpha"),
        finite(target_power, "target_power"),
    )
    if np.any(
        (n < 1)
        | (p0 <= 0)
        | (p0 >= 1)
        | (level <= 0)
        | (level >= 1)
        | (target < level)
        | (target >= 1)
    ):
        raise ValueError("Require positive integral trials, 0<p0<1 and 0<alpha<=target_power<1")
    lower = np.full(n.shape, alternative == "less")
    included = _maximum_region(n, p0, lower, level)
    if np.any(included == 0):
        raise ValueError("An empty critical region cannot attain positive target power")
    critical, size = _binomial_region(n, lower, included, p0)
    if alternative == "less":
        guess = betainccinv(critical + 1, n - critical, target)
    else:
        guess = betaincinv(critical, n - critical + 1, target)
    lo, hi = _probability_bracket(
        n,
        lower,
        included,
        target,
        np.where(lower, 0, p0),
        np.where(lower, p0, 1),
        guess,
        equality_right=alternative == "less",
    )
    pa = np.where(lower, lo, hi)
    # If the requested power already equals the null tail, the closest point is p0.
    pa = np.where(size >= target, p0, pa)
    lo = np.where(size >= target, p0, lo)
    hi = np.where(size >= target, p0, hi)
    _, power = _binomial_region(n, lower, included, pa)
    if np.any(power < target):
        raise ArithmeticError("Returned alternative does not attain target power")
    return BinomialAlternative(n, p0, level, target, alternative, pa, critical, size, power, lo, hi)
