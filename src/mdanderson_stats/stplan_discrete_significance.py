"""Exact one-sided significance planning for STPLAN binomial and Poisson tests.

The inverse is a discrete critical-region search, not a continuous root solve.
For an attainable target, it returns the smallest nonrandomized one-sided tail
whose alternative probability reaches the target. Regions are constrained to
have null probability at most ``STPLAN_MAX_SIGNIFICANCE``. If no such region
reaches the target, the most powerful allowed region is returned with
``target_attained=False``; the native Fortran routine instead reports a
failure status and can leave its power field stale.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import binom, poisson

from ._validation import scalar

STPLAN_MAX_SIGNIFICANCE = 0.99999999
_MAX_BINOMIAL_SIZE = 10_000_000
_MAX_POISSON_MEAN = float(2**48)
_MAX_POISSON_COUNT = 2**53 - 1
_MAX_POISSON_SEARCH_STEPS = 60


@dataclass(frozen=True, slots=True)
class STPLANDiscreteSignificance:
    """A selected nonrandomized one-sided exact critical region.

    ``critical_tail='lower'`` rejects for count ``X <= critical_count``;
    ``'upper'`` rejects for ``X >= critical_count``. A lower count of -1 or
    upper count of ``support_max + 1`` denotes the empty region. The result
    records actual null significance and alternative power, including when
    the requested target cannot be reached under the source significance cap.
    """

    family: Literal["binomial", "poisson"]
    critical_tail: Literal["lower", "upper"]
    critical_count: int
    support_max: int | None
    target_power: float
    significance: float
    achieved_power: float
    target_attained: bool
    significance_limit: float = STPLAN_MAX_SIGNIFICANCE


def _probability(value: float, name: str) -> float:
    result = scalar(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return result


def _result(
    family: Literal["binomial", "poisson"],
    tail: Literal["lower", "upper"],
    critical: int,
    support_max: int | None,
    target: float,
    alpha: float,
    power: float,
) -> STPLANDiscreteSignificance:
    if not np.isfinite(alpha) or not np.isfinite(power):
        raise ArithmeticError("discrete significance calculation is not finite")
    tol = 32 * np.finfo(float).eps
    if alpha < -tol or alpha > 1 + tol or power < -tol or power > 1 + tol:
        raise ArithmeticError("discrete significance calculation is outside [0, 1]")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    power = float(np.clip(power, 0.0, 1.0))
    return STPLANDiscreteSignificance(
        family=family,
        critical_tail=tail,
        critical_count=int(critical),
        support_max=support_max,
        target_power=target,
        significance=alpha,
        achieved_power=power,
        target_attained=power >= target,
    )


def stplan_exact_binomial_significance(
    null_probability: float,
    alternative_probability: float,
    sample_size: int,
    *,
    target_power: float,
) -> STPLANDiscreteSignificance:
    """Invert exact one-sided binomial power by selecting a critical tail.

    The support is bounded to ten million counts so scalar tail searches remain
    predictable. The direction is lower when the alternative is below the
    null, and upper otherwise. Equal hypotheses have no test direction.
    """
    p0 = _probability(null_probability, "null_probability")
    pa = _probability(alternative_probability, "alternative_probability")
    target = _probability(target_power, "target_power")
    n_value = scalar(sample_size, "sample_size")
    if n_value != np.floor(n_value) or not 2 <= n_value <= _MAX_BINOMIAL_SIZE:
        raise ValueError(f"sample_size must be an integer in [2, {_MAX_BINOMIAL_SIZE}]")
    n = int(n_value)
    if p0 == pa:
        raise ValueError("null_probability and alternative_probability must differ")

    if pa < p0:
        # Largest lower cutoff allowed by the null significance cap.
        lo, hi = -1, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if binom.cdf(mid, n, p0) <= STPLAN_MAX_SIGNIFICANCE:
                lo = mid
            else:
                hi = mid
        feasible = lo
        # Power increases with the lower cutoff; pick the first qualifying one.
        left, right = -1, feasible
        while right - left > 1:
            mid = (left + right) // 2
            if mid >= 0 and binom.cdf(mid, n, pa) >= target:
                right = mid
            else:
                left = mid
        critical = (
            right
            if 0 <= right <= feasible and binom.cdf(right, n, pa) >= target
            else feasible
        )
        alpha = 0.0 if critical < 0 else float(binom.cdf(critical, n, p0))
        power = 0.0 if critical < 0 else float(binom.cdf(critical, n, pa))
        return _result("binomial", "lower", critical, n, target, alpha, power)

    # Upper-tail regions are {X >= k}; k=0 is the whole support and exceeds
    # the source significance cap. Find the smallest feasible cutoff first.
    lo, hi = -1, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        alpha_mid = 1.0 if mid <= 0 else float(binom.sf(mid - 1, n, p0))
        if alpha_mid <= STPLAN_MAX_SIGNIFICANCE:
            hi = mid
        else:
            lo = mid
    feasible = hi
    # Power decreases with k; find the largest qualifying cutoff, or keep the
    # smallest allowed one when target power is unattainable.
    left, right = feasible - 1, n + 1
    while right - left > 1:
        mid = (left + right) // 2
        power_mid = 0.0 if mid > n else float(binom.sf(mid - 1, n, pa))
        if power_mid >= target:
            left = mid
        else:
            right = mid
    critical = left if feasible <= left <= n else feasible
    alpha = float(binom.sf(critical - 1, n, p0))
    power = float(binom.sf(critical - 1, n, pa))
    return _result("binomial", "upper", critical, n, target, alpha, power)



def _grow_count(count: int) -> int:
    if count >= _MAX_POISSON_COUNT:
        raise ArithmeticError("Poisson tail did not bracket below the requested probability")
    return min(2 * count + 1, _MAX_POISSON_COUNT)


def _poisson_lower_cutoff_limit(mean: float) -> int:
    # Largest k with CDF(k) <= cap, or -1 for the empty lower region.
    if float(poisson.cdf(0, mean)) > STPLAN_MAX_SIGNIFICANCE:
        return -1
    lo, hi = 0, 1
    for _ in range(_MAX_POISSON_SEARCH_STEPS):
        if float(poisson.cdf(hi, mean)) > STPLAN_MAX_SIGNIFICANCE:
            break
        lo, hi = hi, _grow_count(hi)
    else:
        raise ArithmeticError("could not bracket the lower-tail significance limit")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(poisson.cdf(mid, mean)) <= STPLAN_MAX_SIGNIFICANCE:
            lo = mid
        else:
            hi = mid
    return lo


def _poisson_upper_cutoff_limit(mean: float) -> int:
    # Smallest k with SF(k-1) <= cap; cutoff zero is always excluded.
    lo, hi = 1, 2
    for _ in range(_MAX_POISSON_SEARCH_STEPS):
        if float(poisson.sf(hi - 1, mean)) <= STPLAN_MAX_SIGNIFICANCE:
            break
        lo, hi = hi, _grow_count(hi)
    else:
        raise ArithmeticError("could not bracket the upper-tail significance limit")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(poisson.sf(mid - 1, mean)) <= STPLAN_MAX_SIGNIFICANCE:
            hi = mid
        else:
            lo = mid
    return hi


def _poisson_upper_power_cutoff(feasible: int, mean: float, target: float) -> int | None:
    # Largest upper-tail cutoff reaching target, or None if none does.
    if float(poisson.sf(feasible - 1, mean)) < target:
        return None
    lo, hi = feasible, feasible + 1
    for _ in range(_MAX_POISSON_SEARCH_STEPS):
        if float(poisson.sf(hi - 1, mean)) < target:
            break
        lo, hi = hi, _grow_count(hi)
    else:
        raise ArithmeticError("could not bracket the requested Poisson power")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(poisson.sf(mid - 1, mean)) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def _poisson_lower_power_cutoff(feasible: int, mean: float, target: float) -> int | None:
    # Smallest lower-tail cutoff reaching target, or None if none does.
    if feasible < 0 or float(poisson.cdf(feasible, mean)) < target:
        return None
    lo, hi = -1, feasible
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(poisson.cdf(mid, mean)) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def stplan_exact_poisson_significance(
    null_rate: float,
    alternative_rate: float,
    exposure: float,
    *,
    target_power: float,
) -> STPLANDiscreteSignificance:
    """Invert exact one-sided Poisson power by selecting a critical tail."""
    rate0 = scalar(null_rate, "null_rate")
    ratea = scalar(alternative_rate, "alternative_rate")
    time = scalar(exposure, "exposure")
    target = _probability(target_power, "target_power")
    if rate0 <= 0.0 or ratea <= 0.0 or time <= 0.0:
        raise ValueError("rates and exposure must be positive")
    mean0 = rate0 * time
    meana = ratea * time
    if not np.isfinite(mean0) or not np.isfinite(meana) or max(mean0, meana) >= _MAX_POISSON_MEAN:
        raise ValueError("Poisson means must be finite and below 2**48")
    if mean0 == 0.0 or meana == 0.0:
        raise ArithmeticError("positive rate-times-exposure mean underflowed to zero")
    if rate0 == ratea:
        raise ValueError("null_rate and alternative_rate must differ")

    if ratea < rate0:
        feasible = _poisson_lower_cutoff_limit(mean0)
        target_cutoff = _poisson_lower_power_cutoff(feasible, meana, target)
        critical = feasible if target_cutoff is None else target_cutoff
        alpha = 0.0 if critical < 0 else float(poisson.cdf(critical, mean0))
        power = 0.0 if critical < 0 else float(poisson.cdf(critical, meana))
        return _result("poisson", "lower", critical, None, target, alpha, power)

    feasible = _poisson_upper_cutoff_limit(mean0)
    target_cutoff = _poisson_upper_power_cutoff(feasible, meana, target)
    critical = feasible if target_cutoff is None else target_cutoff
    alpha = float(poisson.sf(critical - 1, mean0))
    power = float(poisson.sf(critical - 1, meana))
    return _result("poisson", "upper", critical, None, target, alpha, power)
