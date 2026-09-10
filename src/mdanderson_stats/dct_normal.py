"""Continuous-endpoint decentralized-trial planning, Tian et al. equations 2.3/2.5."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, ndtr, ndtri

from ._validation import scalar
from .boin import _owned


@dataclass(frozen=True)
class DCTNormalSampleSize:
    unrounded_total: float
    allocation: NDArray[np.int64]
    achieved_power: float
    target_power: float
    alpha: float
    sides: int

    @property
    def total(self) -> int:
        return int(self.allocation.sum())


def dct_normal_sample_size(
    effect: float,
    onsite_sd: float,
    offsite_sd: float,
    *,
    offsite_fraction: float = 0.75,
    relative_bias: float = 0,
    randomization_ratio: float = 1,
    onsite_repeats: int = 1,
    offsite_repeats: int = 1,
    onsite_correlation: float = 0,
    offsite_correlation: float = 0,
    power: float = 0.8,
    alpha: float = 0.05,
    sides: int = 2,
) -> DCTNormalSampleSize:
    """Plan the optimal weighted z-test with equal arm variances within each site.

    effect is the onsite effect; offsite effect is (1+relative_bias)*effect.
    fraction 0/1 selects entirely onsite/offsite data. Allocation rows are onsite,
    offsite; columns are control, experimental. Each continuous arm count is
    rounded upward independently. Repeats are per participant, or cluster sizes
    when allocation is interpreted as numbers of clusters. No dropout adjustment.
    """
    delta = scalar(effect, "effect")
    sd = np.array([scalar(onsite_sd, "onsite_sd"), scalar(offsite_sd, "offsite_sd")])
    fraction = scalar(offsite_fraction, "offsite_fraction")
    bias = scalar(relative_bias, "relative_bias")
    ratio = scalar(randomization_ratio, "randomization_ratio")
    target, error = scalar(power, "power"), scalar(alpha, "alpha")
    m = np.array(
        [scalar(onsite_repeats, "onsite_repeats"), scalar(offsite_repeats, "offsite_repeats")]
    )
    rho = np.array(
        [
            scalar(onsite_correlation, "onsite_correlation"),
            scalar(offsite_correlation, "offsite_correlation"),
        ]
    )
    if delta <= 0 or np.any(sd <= 0) or not 0 <= fraction <= 1 or bias <= -1:
        raise ValueError("require positive effect/SDs, fraction in [0,1], relative_bias > -1")
    if not 1e-6 <= ratio <= 1e6 or sides not in (1, 2) or not 0 < error < target < 1:
        raise ValueError("require ratio in [1e-6,1e6], sides 1/2, and 0 < alpha < power < 1")
    if np.any((m < 1) | (m > 1000000) | (m != np.floor(m))) or np.any((rho < 0) | (rho > 1)):
        raise ValueError("require integer repeats in [1,1000000] and correlations in [0,1]")
    factors = rho + (1 - rho) / m
    fractions = np.array([1 - fraction, fraction])
    arm_fractions = np.array([1 / (1 + ratio), ratio / (1 + ratio)])
    log_effect = np.log(delta) + [0, np.log1p(bias)]
    log_information = 2 * (log_effect - np.log(sd)) - np.log(factors)
    with np.errstate(divide="ignore"):
        log_rate = logsumexp(np.log(fractions) + log_information) + np.log(arm_fractions).sum()
    critical = -float(ndtri(error / sides))
    shift = critical + float(ndtri(target))
    if shift <= 0:
        raise ValueError("requested power is outside the positive-effect planning approximation")
    required = float(np.exp(2 * np.log(shift) - log_rate))
    if not np.isfinite(required) or not 0 < required <= 1e9:
        raise ArithmeticError("sample size is not representable or exceeds one billion units")
    allocation = np.ceil(required * fractions[:, None] * arm_fractions).astype(np.int64)
    active = fractions > 0
    n = allocation[active].astype(float)
    # Recompute information after independent rounding changes allocation ratios.
    log_actual = log_information[active] - np.log(1 / n[:, 0] + 1 / n[:, 1])
    noncentrality = float(np.exp(0.5 * logsumexp(log_actual)))
    attained = float(ndtr(noncentrality - critical))
    if sides == 2:
        attained += float(ndtr(-noncentrality - critical))
    return DCTNormalSampleSize(required, _owned(allocation), attained, target, error, int(sides))
