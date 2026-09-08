"""Boundary-selection probabilities for KSBIN1's stage-design table."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .kstage_binomial import KStageBinomial
from .onesample import binomial_test


@dataclass(frozen=True)
class KSBinomialBoundaryTable:
    """Rows use ascending event counts on the last axis, in either direction.

    significance and power are this stage's rejection contributions. power_loss
    is the probability of reaching the proposed futility region and subsequently
    rejecting in the reference single-stage trial, ignoring future stopping rules.
    It is an absolute probability, not a fraction of reference power.
    """

    design: KStageBinomial
    stage: int
    alternative: str
    single_stage_critical: int
    null_probability: FloatArray
    alternative_probability: FloatArray
    events: np.ndarray
    significance: FloatArray
    power: FloatArray
    conditional_reference_power: FloatArray
    power_contribution: FloatArray
    power_loss: FloatArray
    single_stage_significance: FloatArray
    single_stage_power: FloatArray


def ksbin1_boundary_table(
    design: KStageBinomial,
    stage: int,
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    single_stage_critical: int,
    *,
    alternative: str = "less",
) -> KSBinomialBoundaryTable:
    """Evaluate every inclusive rejection/futility cutoff at one stage.

    Only boundaries before stage affect its arrival distribution. Future stages
    may have disabled boundaries while a design is being constructed. The reference
    single-stage trial uses the design's final sample size and the supplied cutoff.
    Null/alternative probabilities broadcast and may include endpoints.
    """
    if not isinstance(design, KStageBinomial):
        raise ValueError("design must be a KStageBinomial")
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be less or greater")
    p0, pa = np.broadcast_arrays(
        finite(null_probability, "null_probability"),
        finite(alternative_probability, "alternative_probability"),
    )
    if np.any((p0 < 0) | (p0 > 1) | (pa < 0) | (pa > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    if np.any(pa >= p0) if alternative == "less" else np.any(pa <= p0):
        raise ValueError("alternative_probability must be on the specified side of the null")
    n = design.cumulative_trials[-1]
    cut = count(single_stage_critical, "single_stage_critical")
    if cut.ndim != 0 or cut > n:
        raise ValueError("single_stage_critical must be an integer in [0, final trials]")
    cutoff = int(cut)
    null_mass = design.stage_distribution(stage, p0)
    alt_mass = design.stage_distribution(stage, pa)
    events = np.arange(null_mass.shape[-1])
    remaining = n - design.cumulative_trials[stage - 1]
    needed = cutoff - events
    if remaining:
        tails = binomial_test(np.clip(needed, 0, remaining), remaining, pa[..., None])
        if alternative == "less":
            conditional = np.where(needed < 0, 0, np.where(needed >= remaining, 1, tails.p_less))
        else:
            conditional = np.where(needed <= 0, 1, np.where(needed > remaining, 0, tails.p_greater))
    else:
        reject = events <= cutoff if alternative == "less" else events >= cutoff
        conditional = np.broadcast_to(reject.astype(float), alt_mass.shape).copy()
    contribution = alt_mass * conditional

    def cumulative(values: FloatArray, reverse: bool) -> FloatArray:
        if reverse:
            return np.cumsum(values[..., ::-1], axis=-1)[..., ::-1]
        return np.cumsum(values, axis=-1)

    less = alternative == "less"
    reference_null = binomial_test(cutoff, n, p0)
    reference_alt = binomial_test(cutoff, n, pa)
    return KSBinomialBoundaryTable(
        design,
        stage,
        alternative,
        cutoff,
        p0,
        pa,
        events,
        cumulative(null_mass, not less),
        cumulative(alt_mass, not less),
        conditional,
        contribution,
        cumulative(contribution, less),
        reference_null.p_less if less else reference_null.p_greater,
        reference_alt.p_less if less else reference_alt.p_greater,
    )
