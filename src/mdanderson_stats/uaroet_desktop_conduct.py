"""Saved-study inference and explicit current UAROET conduct policies."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _real_matrix
from .uaroet_desktop_rules import (
    UAROETCohortState,
    _profile,
    uaroet_desktop_assignment,
    uaroet_desktop_screen,
    uaroet_replay_cohorts,
    uaroet_utility_cutoff,
)
from .uaroet_desktop_study import UAROETDesktopStudy
from .uaroet_fit import UAROETFit
from .uaroet_precision import UAROETPrecisionFit, fit_uaroet_desktop


@dataclass(frozen=True)
class UAROETDesktopDecision:
    action: str
    selected: int | None
    reason: str
    profile: str
    final: bool
    status: tuple[int, ...]
    allocation_weights: FloatArray
    utility_cutoff: float | None
    posterior: UAROETPrecisionFit | None
    cohort_state: UAROETCohortState
    study: UAROETDesktopStudy


def recommend_uaroet_desktop(
    study: UAROETDesktopStudy,
    *,
    rng: np.random.Generator,
    profile: str = "safe",
    final: bool = False,
    cohort_state: UAROETCohortState | None = None,
    target: float | None = None,
    chunk_draws: int = 1000,
    max_draws: int = 100_000,
    max_evaluations: int = 2_000_000,
    conduct: bool = True,
    maximum_rhat: float = 1.05,
    on_chunk: Callable[[int, int, UAROETFit], None] | None = None,
) -> UAROETDesktopDecision:
    """Fit the exact current study before a precision-qualified recommendation.

    Safe mode screens first-cohort patients whenever observed data are present,
    never restores toxicity exclusions, and uses corrected cohort counters.
    The explicit legacy_1_8 profile retains source defects and distinct tie rules.
    Precision exhaustion returns a computational hold, not a clinical stop.
    Returned state describes already assigned patients; advance it only after
    an actual assignment. Saved histories and live states remain distinct.
    """
    _profile(profile)
    if not isinstance(study, UAROETDesktopStudy) or not isinstance(rng, np.random.Generator):
        raise ValueError("require a study and explicit NumPy Generator")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if not isinstance(conduct, (bool, np.bool_)):
        raise ValueError("conduct must be boolean")
    rhat_limit = _real_matrix(maximum_rhat, "maximum_rhat")
    if rhat_limit.ndim or float(rhat_limit) < 1:
        raise ValueError("maximum_rhat must be finite and at least one")
    maximum_rhat = float(rhat_limit)
    if on_chunk is not None and not callable(on_chunk):
        raise ValueError("on_chunk must be callable")
    p = study.trial_parameters
    D = study.dose_count
    check_within = bool(p.get("CheckAcceptabilityWithinCohort", 1))
    if profile == "safe" and not check_within:
        raise ValueError("within-cohort bypass requires the legacy profile")
    n = len(study.patients)
    if final and n == 0:
        raise ValueError("final selection requires assigned patients")
    if not final and n >= p["NumPatients"]:
        raise ValueError("maximum accrual reached; request final selection")
    state = cohort_state
    if state is None:
        state = uaroet_replay_cohorts(
            tuple(row[1] for row in study.patients),
            dose_count=D,
            new_size=int(p["CohortSize_NewDose"]),
            old_size=int(p["CohortSize_OldDose"]),
            profile=profile,
        )
    elif not isinstance(state, UAROETCohortState) or state.total != n or len(state.first_seen) != D:
        raise ValueError("cohort state must match the supplied history")
    expected = [0] * D
    for index, row in enumerate(study.patients, 1):
        if not expected[row[1]]:
            expected[row[1]] = index
    if tuple(expected) != state.first_seen or (
        (state.new_open or state.old_open) and n and state.last != study.patients[-1][1]
    ):
        raise ValueError("cohort state does not correspond to patient assignments")
    first = int(p["CohortSize_First"])
    start = int(p["DoseStart"]) - 1
    w = np.zeros(D)
    if not final and n < first and (profile == "legacy_1_8" or n == 0):
        w[start] = 1
        return UAROETDesktopDecision(
            "treat",
            start,
            "initial_cohort",
            profile,
            False,
            (),
            _freeze(w),
            None,
            None,
            state,
            study,
        )
    fit = fit_uaroet_desktop(
        study,
        rng=rng,
        target=target,
        conduct=conduct,
        on_chunk=on_chunk,
        chunk_draws=chunk_draws,
        max_draws=max_draws,
        max_evaluations=max_evaluations,
    )
    if not fit.precision_met:
        return UAROETDesktopDecision(
            "hold",
            None,
            "posterior_precision_not_met",
            profile,
            final,
            (),
            _freeze(w),
            None,
            fit,
            state,
            study,
        )
    if fit.utility_samples.shape[0] > 1 and (
        not np.all(np.isfinite(fit.utility_split_rhat))
        or np.max(fit.utility_split_rhat) > maximum_rhat
    ):
        return UAROETDesktopDecision(
            "hold",
            None,
            "posterior_chains_disagree",
            profile,
            final,
            (),
            _freeze(w),
            None,
            fit,
            state,
            study,
        )
    c = study.counts()
    delta = uaroet_utility_cutoff(
        planned=int(p["NumPatients"]),
        steps=int(p["NumTimesToShrinkUtilityCutoff"]),
        observed=int(c.likelihood.sum()),
        initial=p["InitialUtilityCutoffToRandomize"],
        final=p["FinalUtilityCutoffToRandomize"],
        profile=profile,
    )
    means = np.einsum("det,et->d", fit.mean_joint, study.utility)
    good = np.einsum("det,et->d", fit.mean_joint, study.utility >= p["MinUtilityGoodSet"])
    bad_toxicity = c.complete_toxicity[:, int(p["LowestUnacceptableToxLevel"]) :].sum(1)
    bad_efficacy = c.complete_efficacy[:, : int(p["LowestAcceptableEffLevel"])].sum(1)
    status = uaroet_desktop_screen(
        fit.toxicity_exceedance,
        fit.efficacy_shortfall,
        means,
        fit.probability_best,
        bad_toxicity=bad_toxicity,
        bad_efficacy=bad_efficacy,
        complete=c.complete,
        highest_allowed=c.legacy_highest_allowed,
        toxicity_cutoff=p["UpperAcceptablePiToxCutoff"],
        efficacy_cutoff=p["UpperAcceptablePiEffCutoff"],
        utility_delta=delta,
        minimum_best=p["MinProbUtilityIsHighest"],
        filter_utility=bool(p["UseHybridGreedyArandAssignment"]),
        profile=profile,
    )
    if not final:
        forced = None
        if n < first and status[start] == 1:
            forced = start
        elif (
            any(v == 1 for v in status)
            and (state.new_open or state.old_open)
            and (not check_within or status[state.last] == 1)
        ):
            forced = state.last
        if forced is not None:
            w[forced] = 1
            return UAROETDesktopDecision(
                "treat",
                forced,
                "open_cohort",
                profile,
                False,
                status,
                _freeze(w),
                delta,
                fit,
                state,
                study,
            )
    assignment = uaroet_desktop_assignment(
        status,
        means,
        good,
        c.assigned,
        hybrid=True,
        mean_weight=bool(p["RandomizeByMeanUtility"]),
        surplus=int(p["NumPatientSurplusBeforeRandomize"]),
        final=final or not bool(p["UseHybridGreedyArandAssignment"]),
        uniform=float(rng.random()),
        profile=profile,
    )
    action = (
        ("select" if final else "treat")
        if assignment.selected is not None
        else ("hold" if assignment.reason == "zero_eligible_weight" else "stop")
    )
    return UAROETDesktopDecision(
        action,
        assignment.selected,
        assignment.reason,
        profile,
        final,
        status,
        assignment.weights,
        delta,
        fit,
        state,
        study,
    )
