"""Recovered Multc Lean compact boundaries for its legacy duration simulation."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .multc_core import MultcLeanDesign
from .multc_legacy_duration import (
    MultcLegacyDuration,
    MultcLegacyPriorDecision,
    _run_multc_legacy_duration,
)


@dataclass(frozen=True)
class MultcLegacyBoundaries:
    """Earliest stopping sizes indexed by responses and NONtoxicities.

    A vector ``(0,)`` rejects the design before enrollment. Every other vector
    ends with the enrollment cap: that terminal entry can be a display
    placeholder, rather than a posterior crossing. Legacy duration replay
    exits at the cap before checking actual stopping rules.
    """

    max_subjects: int
    min_subjects: int
    cohort_size: int
    response_stop_at: tuple[int, ...]
    nontoxicity_stop_at: tuple[int, ...]

    @property
    def prior_response(self) -> bool:
        return self.response_stop_at == (0,)

    @property
    def prior_toxicity(self) -> bool:
        return self.nontoxicity_stop_at == (0,)

    @property
    def prior_rejected(self) -> bool:
        return self.prior_response or self.prior_toxicity


def multc_legacy_boundaries(design: MultcLeanDesign) -> MultcLegacyBoundaries:
    """Convert verified marginal posterior rules to native compact vectors.

    The native prior screen precedes minimum enrollment, and is mandatory:
    designs with ``pretrial_check=False`` are rejected. Remaining comparisons
    use the design's already calculated posterior bounds, with strict cutoff
    ties and its cohort/minimum schedule. At each look the native loop tests
    only success counts strictly below the treated count. Toxicity successes
    mean NONtoxicities. Cap entries have the native placeholder semantics.
    """
    if not isinstance(design, MultcLeanDesign):
        raise ValueError("design must be a MultcLeanDesign")
    if not design.pretrial_check:
        raise ValueError("legacy boundaries require pretrial_check=True")
    cap = design.max_subjects
    bounds = design.stopping_bounds()

    def vector(endpoint: str) -> tuple[int, ...]:
        if design._stop(endpoint, design._prob(endpoint, 0, 0)):
            return (0,)
        values: list[int] = []
        for index, look in enumerate(bounds.looks):
            n = int(look)
            limit = (
                int(bounds.response_stop_max[index])
                if endpoint == "response"
                else n - int(bounds.toxicity_stop_min[index])
            )
            # Original loop advances a single success index, never evaluating
            # the all-success count i == n or revisiting earlier indices.
            limit = min(limit, n - 1)
            if limit >= len(values):
                values.extend([n] * (limit + 1 - len(values)))
        if not values or (len(values) < cap and values[-1] < cap):
            values.append(cap)
        return tuple(values)

    return MultcLegacyBoundaries(
        cap, design.min_subjects, design.cohort_size, vector("response"), vector("toxicity")
    )


def run_multc_legacy_design_duration(
    design: MultcLeanDesign,
    *,
    joint_probabilities: ArrayLike,
    mean_interarrival: float,
    response_window: float,
    uniforms: ArrayLike,
    unit_exponentials: ArrayLike,
) -> MultcLegacyDuration:
    """Replay a design with recovered boundary, pretrial and duration behavior.

    A prior rejection returns zero patients, duration and consumed draws, with
    decision ``prior_response``, ``prior_toxicity`` or ``prior_both``. All input
    arrays/times are validated even when no patients can enroll. Otherwise the
    recovered compact vectors feed the explicit-variate legacy kernel. Native
    random streams and Windows file/report formatting are not reproduced.
    """
    boundaries = multc_legacy_boundaries(design)
    prior_decision: MultcLegacyPriorDecision | None = None
    if boundaries.prior_response and boundaries.prior_toxicity:
        prior_decision = "prior_both"
    elif boundaries.prior_response:
        prior_decision = "prior_response"
    elif boundaries.prior_toxicity:
        prior_decision = "prior_toxicity"
    return _run_multc_legacy_duration(
        boundaries.max_subjects,
        response_stop_at=() if boundaries.prior_response else boundaries.response_stop_at,
        nontoxicity_stop_at=() if boundaries.prior_toxicity else boundaries.nontoxicity_stop_at,
        joint_probabilities=joint_probabilities,
        mean_interarrival=mean_interarrival,
        response_window=response_window,
        uniforms=uniforms,
        unit_exponentials=unit_exponentials,
        prior_decision=prior_decision,
    )
