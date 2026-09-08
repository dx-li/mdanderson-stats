"""Operating characteristics of fixed KSBIN1 multistage binomial designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .kstage_binomial import KStageBinomial


@dataclass(frozen=True)
class KSBinomialOperatingCharacteristics:
    """Per-stage probabilities use the last axis; leading axes follow probability.

    Conditional expected sample sizes are NaN when the conditioning decision has
    zero probability. expected_sample_size is unconditional. Quitting means
    terminating without rejection, including all nonrejections at the final stage.
    """

    design: KStageBinomial
    alternative: str
    critical: tuple[int, ...]
    probability: FloatArray
    rejection: FloatArray
    quitting: FloatArray
    continuation: FloatArray
    rejection_probability: FloatArray
    expected_sample_size: FloatArray
    expected_given_rejection: FloatArray
    expected_given_quitting: FloatArray


def ksbin1_operating_characteristics(
    cumulative_trials: ArrayLike,
    critical: ArrayLike,
    quit: ArrayLike,
    probability: ArrayLike,
    *,
    alternative: str = "less",
) -> KSBinomialOperatingCharacteristics:
    """Evaluate a fixed KSBIN1 design, with inclusive rejection/quit cutoffs.

    critical contains a count for every stage; quit contains interim counts only.
    -1 disables a boundary. For alternative='less', reject at or below critical
    and quit at or above quit; 'greater' reverses these inequalities. All final
    nonrejections quit. The underlying design permits 1–10 stages and <=200 trials.
    probability may be an array, e.g. [null_probability, alternative_probability].
    """
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be less or greater")
    totals = count(cumulative_trials, "cumulative_trials")
    cuts = finite(critical, "critical")
    if (
        totals.ndim != 1
        or cuts.shape != totals.shape
        or np.any(cuts != np.floor(cuts))
        or np.any(cuts < -1)
        or np.any(cuts > totals)
    ):
        raise ValueError("critical requires one integer in [-1, stage total] per stage")
    low, high = (cuts[:-1], quit) if alternative == "less" else (quit, cuts[:-1])
    design = KStageBinomial(totals, low, high)
    p = finite(probability, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability must lie in [0,1]")
    shape = (*p.shape, len(totals))
    rejection, quitting, continuation = (np.zeros(shape) for _ in range(3))
    for i, n in enumerate(design.cumulative_trials):
        mass = design.stage_distribution(i + 1, p)
        k = np.arange(n + 1)
        reject = (k <= cuts[i]) if alternative == "less" else (k >= cuts[i])
        reject &= cuts[i] != -1
        if i == len(totals) - 1:
            stop = ~reject
        else:
            stop = (k >= design.high[i]) if alternative == "less" else (k <= design.low[i])
            boundary = design.high[i] if alternative == "less" else design.low[i]
            stop &= boundary != -1
        rejection[..., i] = np.sum(mass[..., reject], axis=-1)
        quitting[..., i] = np.sum(mass[..., stop], axis=-1)
        continuation[..., i] = np.sum(mass[..., ~(reject | stop)], axis=-1)
    total_rejection = np.sum(rejection, axis=-1)
    total_quitting = np.sum(quitting, axis=-1)
    weighted_rejection = np.sum(rejection * totals, axis=-1)
    weighted_quitting = np.sum(quitting * totals, axis=-1)
    conditional_rejection = np.full(p.shape, np.nan)
    conditional_quitting = np.full(p.shape, np.nan)
    np.divide(
        weighted_rejection, total_rejection, out=conditional_rejection, where=total_rejection > 0
    )
    np.divide(weighted_quitting, total_quitting, out=conditional_quitting, where=total_quitting > 0)
    return KSBinomialOperatingCharacteristics(
        design,
        alternative,
        tuple(map(int, cuts)),
        p,
        rejection,
        quitting,
        continuation,
        total_rejection,
        weighted_rejection + weighted_quitting,
        conditional_rejection,
        conditional_quitting,
    )
