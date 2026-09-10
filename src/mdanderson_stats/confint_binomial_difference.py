"""CONFINT assurance for the unadjusted two-binomial Wald interval width."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import binom

from ._validation import scalar
from .bayesian_monitoring import _integer
from .confint_binomial import _assurance


def confint_binomial_difference_probability(
    sample_size1: int,
    sample_size2: int,
    event_probability1: float,
    event_probability2: float,
    max_length: float,
    *,
    confidence: float = 0.95,
) -> float:
    """Probability the plain two-sided Wald interval has total length <=L.

    Independent samples; observed unpooled binomial variances. No continuity
    correction, boundary adjustment or clipping. This is width assurance, not
    a guarantee of nominal CI coverage. Enumerates half of the smaller sample
    and evaluates the other sample's qualifying counts with CDF/SF tails.
    """
    n1, n2 = _integer(sample_size1, "sample_size1"), _integer(sample_size2, "sample_size2")
    p1 = scalar(event_probability1, "event_probability1")
    p2 = scalar(event_probability2, "event_probability2")
    length, level = scalar(max_length, "max_length"), scalar(confidence, "confidence")
    if not 1 <= n1 <= 1000000 or not 1 <= n2 <= 1000000:
        raise ValueError("sample sizes must be integers in 1..1000000")
    if not 0 <= p1 <= 1 or not 0 <= p2 <= 1 or length <= 0:
        raise ValueError("require probabilities in [0,1] and positive max_length")
    if not 1e-6 <= level <= 1 - 1e-12:
        raise ValueError("confidence must be in [1e-6,1-1e-12]")
    if n2 < n1:
        n1, n2, p1, p2 = n2, n1, p2, p1
    z = -2 * float(ndtri((1 - level) / 2))
    maximum = (n1 // 2) * (n1 - n1 // 2) / n1**3 + (n2 // 2) * (n2 - n2 // 2) / n2**3
    if length >= z * np.sqrt(maximum):
        return 1.0
    # The previous comparison bounds length/z, so its square cannot overflow.
    budget = (length / z) ** 2
    counts = np.arange(n1 // 2 + 1, dtype=float)
    mass = binom.pmf(counts, n1, p1)
    mirror = n1 - counts
    mass += np.where(mirror != counts, binom.pmf(mirror, n1, p1), 0)
    remaining = budget - counts * (n1 - counts) / n1**3
    possible = remaining >= 0
    variance = np.maximum(remaining, 0)
    # Smaller root of p*(1-p)=n2*variance, with no 1-sqrt cancellation.
    discriminant = np.maximum(1 - 4 * n2 * variance, 0)
    root = (2 * n2 * variance) / (1 + np.sqrt(discriminant))
    cutoff = np.minimum(np.floor(n2 * root), n2 // 2)
    too_high = cutoff * (n2 - cutoff) / n2**3 > variance
    cutoff -= too_high
    next_count = cutoff + 1
    too_low = (next_count <= n2 // 2) & (next_count * (n2 - next_count) / n2**3 <= variance)
    cutoff += too_low
    conditional = binom.cdf(cutoff, n2, p2) + binom.sf(n2 - cutoff - 1, n2, p2)
    conditional[cutoff >= n2 // 2] = 1
    conditional[~possible] = 0
    return float(np.clip(mass @ conditional, 0, 1))


def confint_binomial_difference_event_limit(
    sample_size1: int,
    sample_size2: int,
    event_probability1: float,
    max_length: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
) -> float | None:
    """Largest p2 in [0,.5] attaining assurance, with symmetric region near 1.

    None means no p2 qualifies; .5 means every p2 qualifies. Group 1 is fixed.
    """
    target = _assurance(assurance)

    def crossing(p: float) -> float:
        return (
            confint_binomial_difference_probability(
                sample_size1, sample_size2, event_probability1, p, max_length, confidence=confidence
            )
            - target
        )

    if crossing(0) < 0:
        return None
    if crossing(0.5) >= 0:
        return 0.5
    return float(brentq(crossing, 0, 0.5, xtol=1e-14))


def confint_binomial_difference_sample_size(
    max_length: float,
    event_probability1: float,
    event_probability2: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
    min_sample_size: int = 10,
    max_sample_size: int = 1000,
) -> int:
    """First qualifying equal per-group n in the requested inclusive range.

    Defaults follow the native 10..1000 search range. Scans every integer since
    discrete assurance can decrease between adjacent sizes. Small-n Wald CIs
    can be misleadingly narrow; this does not repair their statistical coverage.
    """
    target = _assurance(assurance)
    lower = _integer(min_sample_size, "min_sample_size")
    upper = _integer(max_sample_size, "max_sample_size")
    if not 1 <= lower <= upper <= 10000:
        raise ValueError("require 1 <= min_sample_size <= max_sample_size <= 10000")
    for n in range(lower, upper + 1):
        probability = confint_binomial_difference_probability(
            n, n, event_probability1, event_probability2, max_length, confidence=confidence
        )
        if probability >= target:
            return n
    raise ValueError("requested assurance is not attained within the sample-size range")
