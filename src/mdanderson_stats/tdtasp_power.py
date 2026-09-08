"""Discrete binomial power under TDTASP's mean-contribution model."""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom

from ._validation import count, scalar
from .binomial_design import _binomial_region, _maximum_region
from .tdtasp_ascertainment import TDTASPAscertainment
from .tdtasp_genetics import FloatArray, _freeze


@dataclass(frozen=True)
class TDTASPFixedPower:
    """Inclusive lower/upper cutoffs; -1 and n+1 denote absent tails."""

    observations: FloatArray
    answer_probability: float
    alpha: float
    sides: int
    legacy_two_sided: bool
    lower_critical: FloatArray
    upper_critical: FloatArray
    actual_size: FloatArray
    power: FloatArray


def tdtasp_fixed_power(
    observations: ArrayLike,
    answer_probability: float,
    alpha: float = 0.05,
    *,
    sides: int = 1,
    legacy_two_sided: bool = False,
) -> TDTASPFixedPower:
    """Nonrandomized test of p=0.5 for scalar or array integral observations.

    One-sided direction follows answer_probability (lower at equality). Two
    sides use both equal-alpha tails; legacy_two_sided retains only the source's
    chosen tail at alpha/2. Zero observations have empty rejection regions.
    """
    n = count(observations, "observations")
    p = scalar(answer_probability, "answer_probability")
    level = scalar(alpha, "alpha")
    if not 0 <= p <= 1 or not 0 < level < 1:
        raise ValueError("answer_probability must lie in [0,1] and alpha in (0,1)")
    if (
        isinstance(sides, (bool, np.bool_))
        or not isinstance(sides, (int, np.integer))
        or sides not in (1, 2)
    ):
        raise ValueError("sides must be 1 or 2")
    if not isinstance(legacy_two_sided, (bool, np.bool_)):
        raise ValueError("legacy_two_sided must be boolean")
    lower, upper = np.full(n.shape, -1.0), np.array(n + 1)
    size, power = np.zeros(n.shape), np.zeros(n.shape)
    active = n > 0
    if np.any(active):
        na = n[active]
        lower_direction = np.full(na.shape, p <= 0.5)
        included = _maximum_region(
            na, np.full(na.shape, 0.5), lower_direction, np.full(na.shape, level / sides)
        )
        cutoff, achieved = _binomial_region(na, lower_direction, included, np.full(na.shape, 0.5))
        _, probability = _binomial_region(na, lower_direction, included, np.full(na.shape, p))
        if sides == 2 and not legacy_two_sided:
            lower[active], upper[active] = included - 1, na - included + 1
            _, opposite = _binomial_region(na, ~lower_direction, included, np.full(na.shape, p))
            size[active], power[active] = 2 * achieved, probability + opposite
        else:
            lower[active] = np.where(lower_direction, cutoff, -1)
            upper[active] = np.where(lower_direction, na + 1, cutoff)
            size[active], power[active] = achieved, probability
    return TDTASPFixedPower(
        _freeze(n),
        p,
        level,
        int(sides),
        bool(legacy_two_sided),
        _freeze(lower),
        _freeze(upper),
        _freeze(size),
        _freeze(power),
    )


@dataclass(frozen=True)
class TDTASPPower:
    """Power averaged over the binomial count of eligible screened families."""

    ascertainment: TDTASPAscertainment
    families: int
    contribution_per_family: float
    legacy_scale: bool
    eligible_families: FloatArray
    family_count_probability: FloatArray
    conditional: TDTASPFixedPower
    actual_size: float
    power: float


def tdtasp_power(
    ascertainment: TDTASPAscertainment,
    families: int,
    alpha: float = 0.05,
    *,
    sides: int = 1,
    legacy_scale: bool = False,
    legacy_two_sided: bool = False,
    max_terms: int = 100_001,
) -> TDTASPPower:
    """Average power over K ~ Binomial(families, eligibility probability).

    Observations are round-half-up(K * mean contributions per eligible family),
    as in the source's power_bin1 ANINT. This is a mean-contribution planning
    approximation, not a full model of correlated sibling/parent observations.
    Every eligible-family count is evaluated, with max_terms checked before
    allocation. Deterministic eligibility needs only one term.

    legacy_scale substitutes the source's product of expected informative
    parents and its unselected truncated offspring mean for all-child TDT.
    Other source compatibility choices reside in ascertainment/genetic inputs.
    """
    if not isinstance(ascertainment, TDTASPAscertainment):
        raise TypeError("ascertainment must be a TDTASPAscertainment")
    try:
        n, budget = index(families), index(max_terms)
    except TypeError as exc:
        raise ValueError("families and max_terms must be integers") from exc
    if isinstance(families, (bool, np.bool_)) or not 0 <= n < 2**53:
        raise ValueError("families must be a nonnegative integer smaller than 2**53")
    if isinstance(max_terms, (bool, np.bool_)) or budget < 1:
        raise ValueError("max_terms must be a positive integer")
    if not isinstance(legacy_scale, (bool, np.bool_)):
        raise ValueError("legacy_scale must be boolean")
    q = ascertainment.selection_probability
    if not 0 < q <= 1:
        raise ArithmeticError("selection probability is outside the representable interval (0,1]")
    scale = ascertainment.expected_contributions
    if legacy_scale and ascertainment.test == "tdt" and ascertainment.all_affected:
        scale = (
            ascertainment.expected_heterozygous_parents
            * ascertainment.population_average_truncated_mean
        )
    eligible, weights = _family_count_distribution(n, q, budget)
    observations = _rounded_observations(eligible, scale)
    conditional = tdtasp_fixed_power(
        observations,
        ascertainment.answer_probability,
        alpha,
        sides=sides,
        legacy_two_sided=legacy_two_sided,
    )
    return TDTASPPower(
        ascertainment,
        n,
        scale,
        bool(legacy_scale),
        _freeze(eligible),
        _freeze(weights),
        conditional,
        float(weights @ conditional.actual_size),
        float(weights @ conditional.power),
    )


def _family_count_distribution(n: int, q: float, budget: int) -> tuple[FloatArray, FloatArray]:
    if q == 1 or n == 0:
        eligible, weights = np.array([float(n)]), np.ones(1)
    else:
        if n + 1 > budget:
            raise ValueError(f"mixture requires {n + 1} terms, exceeding max_terms={budget}")
        eligible = np.arange(n + 1, dtype=float)
        weights = binom.pmf(eligible, n, q)
        total = float(np.sum(weights, dtype=np.longdouble))
        if not np.isfinite(total) or abs(total - 1) > 1e-10:
            raise ArithmeticError("binomial family-count probabilities do not sum to one")
        weights /= total
    return eligible, weights


def _rounded_observations(eligible: FloatArray, scale: float) -> FloatArray:
    raw_observations = eligible * scale
    integral = np.floor(raw_observations)
    observations = integral + (raw_observations - integral >= 0.5)
    if np.any(~np.isfinite(observations)) or np.any(observations >= 2**53):
        raise ValueError("rounded observation counts exceed the exact integer range")
    return observations
