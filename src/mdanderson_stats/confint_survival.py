"""CONFINT exponential-survival CI-width assurance under its gamma approximation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv
from scipy.stats import poisson

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .stattab_probability import stattab_poisson_term
from .survival_sample_size import exponential_event_probability


@dataclass(frozen=True)
class CONFINTSurvivalAssurance:
    probability: float
    expected_events: float
    event_probability: float
    omitted_probability: float
    included_events: tuple[int, int] | None


def confint_survival_fixed_events(
    events: ArrayLike,
    hazard: ArrayLike,
    max_length: ArrayLike,
    *,
    confidence: ArrayLike = 0.95,
    target: str = "hazard",
) -> FloatArray:
    """Width probability conditional on n events, using T ~ Gamma(n, rate=hazard).

    target is hazard or mean; max_length is total CI length in corresponding
    units. The source assigns zero probability to n=0 (undefined interval).
    Inputs broadcast; outputs are read-only. This gamma model is approximate
    for general censoring, rather than an exact conditional calendar-trial law.
    """
    n, h, length, level = np.broadcast_arrays(
        count(events, "events"),
        finite(hazard, "hazard"),
        finite(max_length, "max_length"),
        finite(confidence, "confidence"),
    )
    if target not in {"hazard", "mean"}:
        raise ValueError("target must be hazard or mean")
    if not 0 < n.size <= 2000000 or np.any(n > 200000000):
        raise ValueError("require 1..2000000 designs with events in 0..200000000")
    if np.any(h <= 0) or np.any(length <= 0) or np.any((level < 1e-6) | (level > 1 - 1e-12)):
        raise ValueError("require positive hazard/length and confidence in [1e-6,1-1e-12]")
    result = np.zeros(n.shape)
    positive = n > 0
    shape, tail = n[positive], (1 - level[positive]) / 2
    lo, hi = gammaincinv(shape, tail), gammainccinv(shape, tail)
    log_difference = np.log(hi - lo)
    if target == "hazard":
        log_argument = np.log(h[positive]) + log_difference - np.log(length[positive])
    else:
        log_argument = (
            np.log(h[positive])
            + np.log(length[positive])
            + np.log(lo)
            + np.log(hi)
            - log_difference
        )
    with np.errstate(over="ignore", under="ignore"):
        argument = np.exp(log_argument)
    result[positive] = (
        gammaincc(shape, argument) if target == "hazard" else gammainc(shape, argument)
    )
    return _owned(result)


def confint_survival_probability(
    hazard: float,
    accrual_rate: float,
    accrual_time: float,
    followup_time: float,
    max_length: float,
    *,
    confidence: float = 0.95,
    target: str = "hazard",
    tail_tolerance: float = 1e-12,
) -> CONFINTSurvivalAssurance:
    """Average conditional width assurance over Poisson-distributed event counts.

    Poisson accrual, uniform entry over accrual_time, independent exponential
    event times, additional followup_time, no dropout/competing risks. The
    event-count mean is exact under this model; conditional gamma follow-up is
    CONFINT's planning approximation. The unnormalized sum is a lower estimate,
    with omitted_probability bounding only the omitted positive-event terms.
    """
    h, rate, accrual, followup, length, level, tolerance = (
        scalar(value, name)
        for value, name in zip(
            [
                hazard,
                accrual_rate,
                accrual_time,
                followup_time,
                max_length,
                confidence,
                tail_tolerance,
            ],
            [
                "hazard",
                "accrual_rate",
                "accrual_time",
                "followup_time",
                "max_length",
                "confidence",
                "tail_tolerance",
            ],
            strict=True,
        )
    )
    # Validate target/interval settings even for a zero-accrual design.
    confint_survival_fixed_events(0, h, length, confidence=level, target=target)
    if rate < 0 or accrual <= 0 or followup < 0 or not 1e-14 <= tolerance <= 1e-4:
        raise ValueError(
            "require rate>=0, accrual_time>0, followup_time>=0 and tolerance in [1e-14,1e-4]"
        )
    event_probability = float(exponential_event_probability(h, accrual, followup))
    if rate == 0:
        return CONFINTSurvivalAssurance(0, 0, event_probability, 0, None)
    if event_probability < 1e-200:
        # First-order event probability is effectively exact in this regime;
        # log form preserves mean counts even when h*duration underflows.
        log_followup = np.log(followup) if followup > 0 else -np.inf
        log_probability = np.log(h) + np.logaddexp(np.log(accrual) - np.log(2), log_followup)
    else:
        log_probability = np.log(event_probability)
    log_mean = np.log(rate) + np.log(accrual) + log_probability
    if log_mean > np.log(100000000):
        raise ValueError("expected event count exceeds 100000000")
    mean = float(np.exp(log_mean))
    if mean == 0:
        return CONFINTSurvivalAssurance(0, 0, event_probability, 0, None)
    lower = max(1, int(poisson.ppf(tolerance / 2, mean)))
    upper = max(lower, int(poisson.isf(tolerance / 2, mean)))
    events = np.arange(lower, upper + 1)
    if events.size > 200000:
        raise ValueError("Poisson summation requires more than 200000 event counts")
    mass = stattab_poisson_term(events, mean)
    conditional = confint_survival_fixed_events(events, h, length, confidence=level, target=target)
    value = float(mass @ conditional)
    omitted = float(poisson.sf(upper, mean))
    if lower > 1:
        omitted += max(0.0, float(poisson.cdf(lower - 1, mean) - np.exp(-mean)))
    if omitted > tolerance * 1.001:
        raise ArithmeticError("Poisson truncation did not meet the requested tolerance")
    return CONFINTSurvivalAssurance(value, mean, event_probability, omitted, (lower, upper))
