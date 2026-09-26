"""Complete-outcome conduct rules for the model-averaged CRM posterior."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bmacrm import BMACRMPosterior


@dataclass(frozen=True)
class BMACRMDecision:
    """One complete-outcome starting, next-dose, stopping, or MTD decision."""

    action: Literal["start", "treat", "stop", "select_mtd"]
    dose: int | None
    unconstrained_dose: int | None
    no_skip_dose: int | None
    safety_probability: float
    raw_rate_limited: bool
    high_uncertainty: bool
    explanation: str


def _dose_index(value: object, name: str, dose_count: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer dose index")
    numeric = float(raw)
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an integer dose index")
    index = int(numeric)
    if not 0 <= index < dose_count:
        raise ValueError(f"{name} is outside the dose range")
    return index


def _nearest_dose(means: np.ndarray, target: float, eligible: np.ndarray) -> int:
    distance = np.abs(means[eligible] - target)
    minimum = float(np.min(distance))
    # Treat distances within roundoff of an exact tie equally; np.argmin alone
    # can select the higher dose when decimal values have asymmetric encoding.
    tied = eligible[
        np.isclose(distance, minimum, rtol=8 * np.finfo(float).eps, atol=8 * np.finfo(float).eps)
    ]
    return int(tied[0])


def bmacrm_decision(
    posterior: BMACRMPosterior,
    *,
    current_dose: int | None = None,
    starting_dose: int = 0,
    safety_cutoff: float = 0.9,
    final: bool = False,
) -> BMACRMDecision:
    """Apply complete-outcome BMA-CRM start, safety, escalation, and MTD rules.

    Dose indices are zero-based. The first untried level caps the posterior
    mean-to-target recommendation; levels below ``starting_dose`` count as
    already tried. For nonfinal upward moves, any fully observed current or
    intervening toxicity rate above the posterior's target caps the move at
    that first level. Final MTD selection instead applies the three-subject
    eligibility fallback. Pending outcomes and cohort-timing rules are outside
    this function's scope.
    """
    if not isinstance(posterior, BMACRMPosterior):
        raise TypeError("posterior must be a BMACRMPosterior")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    final_value = bool(final)
    means = np.asarray(posterior.dose_mean, dtype=float)
    overdose = np.asarray(posterior.overdose_probability, dtype=float)
    events = np.asarray(posterior.events, dtype=float)
    subjects = np.asarray(posterior.subjects, dtype=float)
    dose_count = means.size
    if (
        means.ndim != 1
        or overdose.shape != means.shape
        or events.shape != means.shape
        or subjects.shape != means.shape
        or dose_count == 0
    ):
        raise ValueError("posterior has inconsistent dose summaries")
    if (
        np.any(~np.isfinite(means))
        or np.any((means < 0) | (means > 1))
        or np.any(~np.isfinite(overdose))
        or np.any((overdose < 0) | (overdose > 1))
        or np.any(~np.isfinite(events))
        or np.any(~np.isfinite(subjects))
        or np.any(events < 0)
        or np.any(subjects < events)
        or np.any(events != np.floor(events))
        or np.any(subjects != np.floor(subjects))
    ):
        raise ValueError("posterior dose summaries/counts are invalid")
    if not np.isfinite(posterior.target) or not 0 < posterior.target < 1:
        raise ValueError("posterior target must lie strictly between 0 and 1")
    start = _dose_index(starting_dose, "starting_dose", dose_count)
    cutoff_array = np.asarray(safety_cutoff)
    if cutoff_array.ndim != 0 or cutoff_array.dtype.kind not in "iuf":
        raise ValueError("safety_cutoff must be a finite scalar in [0,1]")
    cutoff = float(cutoff_array)
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("safety_cutoff must be a finite scalar in [0,1]")

    total_observed = float(np.sum(subjects))
    safety_probability = float(overdose[0])
    unconstrained = _nearest_dose(means, posterior.target, np.arange(dose_count))
    if total_observed == 0:
        if final_value:
            raise ValueError("cannot select a final MTD without observed subjects")
        return BMACRMDecision(
            "start",
            start,
            unconstrained,
            start,
            safety_probability,
            False,
            False,
            "Start at the configured dose before complete outcomes are observed.",
        )

    current: int | None = None
    if current_dose is not None:
        current = _dose_index(current_dose, "current_dose", dose_count)
    if not final_value:
        if current is None:
            raise ValueError("current_dose is required for a noninitial nonfinal decision")
        if subjects[current] < 1:
            raise ValueError("current_dose must have at least one observed subject")

    if cutoff < 1 and safety_probability > cutoff:
        return BMACRMDecision(
            "stop",
            None,
            unconstrained,
            None,
            safety_probability,
            False,
            False,
            "Stop because posterior lowest-dose overdose probability exceeds safety_cutoff.",
        )

    tried = subjects > 0
    tried[:start] = True
    untried = np.flatnonzero(~tried)
    no_skip_limit = int(untried[0]) if untried.size else dose_count - 1
    eligible = np.arange(no_skip_limit + 1)
    no_skip = _nearest_dose(means, posterior.target, eligible)

    if final_value:
        dose = no_skip
        high_uncertainty = False
        if subjects[dose] < 3:
            lower_with_three = np.flatnonzero(subjects[:dose] >= 3)
            if lower_with_three.size:
                dose = int(lower_with_three[-1])
            else:
                high_uncertainty = True
        return BMACRMDecision(
            "select_mtd",
            dose,
            unconstrained,
            no_skip,
            safety_probability,
            False,
            high_uncertainty,
            (
                "Select the nearest eligible dose to target with the three-subject rule."
                if not high_uncertainty
                else "Retain the nearest eligible dose; no lower dose has three subjects."
            ),
        )

    dose = no_skip
    raw_rate_limited = False
    if current is not None and dose > current:
        observed_rate = np.divide(events, subjects, out=np.zeros_like(events), where=subjects > 0)
        excessive = np.flatnonzero(
            (np.arange(dose_count) >= current)
            & (np.arange(dose_count) < dose)
            & (subjects > 0)
            & (observed_rate > posterior.target)
        )
        if excessive.size:
            dose = int(excessive[0])
            raw_rate_limited = True
    explanation = (
        "The recommendation is capped at the first current or intervening dose whose fully "
        "observed toxicity rate exceeds target."
        if raw_rate_limited
        else "Treat at the closest posterior mean-to-target dose permitted by the no-skip rule."
    )
    return BMACRMDecision(
        "treat",
        dose,
        unconstrained,
        no_skip,
        safety_probability,
        raw_rate_limited,
        False,
        explanation,
    )
