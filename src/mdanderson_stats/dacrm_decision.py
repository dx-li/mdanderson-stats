"""Source-profile dose decisions for a fitted DA-CRM posterior."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bmacrm_decision import _nearest_dose
from .dacrm import DACRMPosterior

_POLICIES = {"paper", "crm_suite"}


@dataclass(frozen=True)
class DACRMDecision:
    """One DA-CRM start, treat, wait, stop, or final MTD decision."""

    action: Literal["start", "treat", "wait", "stop", "select_mtd"]
    dose: int | None
    unconstrained_dose: int
    no_skip_dose: int | None
    safety_probability: float
    raw_rate_limited: bool
    high_uncertainty: bool
    policy: Literal["paper", "crm_suite"]
    explanation: str


def _index(value: object, name: str, size: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer dose index")
    numeric = float(raw)
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an integer dose index")
    result = int(numeric)
    if not 0 <= result < size:
        raise ValueError(f"{name} is outside the dose range")
    return result


def _integer(value: object, name: str, low: int, high: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    numeric = float(raw)
    if not np.isfinite(numeric) or numeric != np.floor(numeric) or not low <= numeric <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(numeric)


def _cutoff(value: object, policy: str) -> float:
    if value is None:
        return 0.96 if policy == "paper" else 0.9
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError("safety_cutoff must be a finite scalar in [0,1]")
    result = float(raw)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("safety_cutoff must be a finite scalar in [0,1]")
    return result


def _validate_posterior(posterior: DACRMPosterior) -> tuple[np.ndarray, ...]:
    means = np.asarray(posterior.dose_mean, dtype=float)
    overdose = np.asarray(posterior.overdose_probability, dtype=float)
    doses = np.asarray(posterior.doses, dtype=float)
    outcomes = np.asarray(posterior.outcomes, dtype=float)
    dose_count = np.asarray(posterior.skeleton).size
    if (
        means.ndim != 1
        or means.size != dose_count
        or not 1 <= dose_count <= 20
        or overdose.shape != means.shape
        or doses.ndim != 1
        or outcomes.shape != doses.shape
        or doses.size > 200
    ):
        raise ValueError("posterior contains inconsistent dose summaries or patient data")
    if (
        np.any(~np.isfinite(means))
        or np.any((means < 0) | (means > 1))
        or np.any(~np.isfinite(overdose))
        or np.any((overdose < 0) | (overdose > 1))
        or np.any(~np.isfinite(doses))
        or np.any(doses != np.floor(doses))
        or np.any((doses < 0) | (doses >= dose_count))
        or np.any(~np.isfinite(outcomes))
        or np.any(~np.isin(outcomes, [-1, 0, 1]))
        or not np.isfinite(posterior.target)
        or not 0 < posterior.target < 1
    ):
        raise ValueError("posterior dose summaries or patient data are invalid")
    allocated = np.bincount(doses.astype(np.int64), minlength=dose_count)
    complete_mask = outcomes != -1
    observed = np.bincount(doses[complete_mask].astype(np.int64), minlength=dose_count)
    events = np.bincount(doses[outcomes == 1].astype(np.int64), minlength=dose_count)
    return means, overdose, allocated, observed, events


def dacrm_decision(
    posterior: DACRMPosterior,
    *,
    current_dose: int | None = None,
    starting_dose: int = 0,
    policy: Literal["paper", "crm_suite"] = "paper",
    safety_cutoff: float | None = None,
    minimum_observed: int | None = None,
    final: bool = False,
) -> DACRMDecision:
    """Apply a published paper or CRM Suite dose-allocation profile.

    Dose indices are zero-based. Pending patients count as treated, but only
    completed outcomes contribute to observed counts and raw DLT rates. The
    paper profile steps no more than one adjacent level at interim and uses a
    strict 0.96 lowest-dose overdose stop. CRM Suite also supports its
    minimum-observation wait and final first-untried/three-treated rules.
    """
    if not isinstance(posterior, DACRMPosterior):
        raise TypeError("posterior must be a DACRMPosterior")
    if policy not in _POLICIES:
        raise ValueError("policy must be 'paper' or 'crm_suite'")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    final_value = bool(final)
    means, overdose, allocated, observed, events = _validate_posterior(posterior)
    dose_count = means.size
    start = _index(starting_dose, "starting_dose", dose_count)
    cutoff = _cutoff(safety_cutoff, policy)
    if policy == "paper":
        if minimum_observed is None:
            minimum = 0
        else:
            minimum = _integer(minimum_observed, "minimum_observed", 0, 200)
            if minimum != 0:
                raise ValueError("minimum_observed is only configurable for the CRM Suite profile")
    else:
        if minimum_observed is None:
            raise ValueError("CRM Suite policy requires explicit minimum_observed")
        minimum = _integer(minimum_observed, "minimum_observed", 0, 200)

    total_treated = int(np.sum(allocated))
    safety = float(overdose[0])
    unconstrained = _nearest_dose(means, posterior.target, np.arange(dose_count))
    if total_treated == 0:
        if final_value:
            raise ValueError("cannot select a final MTD without treated patients")
        return DACRMDecision(
            "start",
            start,
            unconstrained,
            start,
            safety,
            False,
            False,
            policy,
            "Start at the configured dose before any patients have been treated.",
        )

    current = None if current_dose is None else _index(current_dose, "current_dose", dose_count)
    if not final_value:
        if current is None:
            raise ValueError("current_dose is required for an interim decision")
        if allocated[current] == 0:
            raise ValueError("current_dose must have at least one treated patient")

    if cutoff < 1 and safety > cutoff:
        return DACRMDecision(
            "stop",
            None,
            unconstrained,
            None,
            safety,
            False,
            False,
            policy,
            "Stop because lowest-dose posterior overdose probability exceeds safety_cutoff.",
        )

    if policy == "paper":
        tried = observed > 0
    else:
        tried = observed > 0 if minimum > 0 else allocated > 0
    tried = tried.copy()
    tried[:start] = True
    untried = np.flatnonzero(~tried)
    first_untried = int(untried[0]) if untried.size else dose_count

    if final_value:
        if policy == "paper":
            no_skip = unconstrained
            selected = unconstrained
            uncertain = False
        else:
            eligible = np.arange(min(first_untried, dose_count - 1) + 1)
            no_skip = _nearest_dose(means, posterior.target, eligible)
            selected = no_skip
            uncertain = False
            if allocated[selected] < 3:
                lower = np.flatnonzero(allocated[:selected] >= 3)
                if lower.size:
                    selected = int(lower[-1])
                else:
                    uncertain = True
        return DACRMDecision(
            "select_mtd",
            selected,
            unconstrained,
            no_skip,
            safety,
            False,
            uncertain,
            policy,
            "Select the policy's final dose recommendation."
            if not uncertain
            else "Retain the recommended dose; no lower dose has three treated patients.",
        )

    assert current is not None
    if unconstrained > current:
        recommendation = min(current + 1, dose_count - 1)
    elif unconstrained < current:
        recommendation = max(current - 1, 0)
    else:
        recommendation = current

    if policy == "paper":
        no_skip = recommendation
    else:
        eligible = np.arange(min(first_untried, dose_count - 1) + 1)
        no_skip = _nearest_dose(means, posterior.target, eligible)
        has_pending = bool(np.any(np.asarray(posterior.outcomes) == -1))
        if recommendation > current and has_pending and observed[current] < minimum:
            return DACRMDecision(
                "wait",
                None,
                unconstrained,
                no_skip,
                safety,
                False,
                False,
                policy,
                "Wait until the current dose has the required number of observed outcomes.",
            )
        if recommendation > current and recommendation > first_untried:
            return DACRMDecision(
                "wait",
                None,
                unconstrained,
                no_skip,
                safety,
                False,
                False,
                policy,
                "Wait because an untried lower dose blocks the upward step.",
            )
        if recommendation > current:
            raw_rates = np.divide(
                events,
                observed,
                out=np.zeros_like(events, dtype=float),
                where=observed > 0,
            )
            excessive = np.flatnonzero(
                (np.arange(dose_count) >= current)
                & (np.arange(dose_count) < recommendation)
                & (observed > 0)
                & (raw_rates > posterior.target)
            )
            if excessive.size:
                recommendation = int(excessive[0])
                return DACRMDecision(
                    "treat",
                    recommendation,
                    unconstrained,
                    no_skip,
                    safety,
                    True,
                    False,
                    policy,
                    "Cap escalation at the first current or intervening dose with raw DLT rate "
                    "above target.",
                )
    return DACRMDecision(
        "treat",
        recommendation,
        unconstrained,
        no_skip,
        safety,
        False,
        False,
        policy,
        "Treat at the dose selected by the requested DA-CRM policy.",
    )
