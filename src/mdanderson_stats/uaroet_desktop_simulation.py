"""Precision-qualified current UAROET trials on an explicit observation calendar."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .uaroet import _integer, _real_matrix
from .uaroet_calendar import (
    UAROETTimedPatient,
    _advance_arrival,
    uaroet_outcome_delays,
    uaroet_study_as_of,
)
from .uaroet_desktop_conduct import UAROETDesktopDecision, recommend_uaroet_desktop
from .uaroet_desktop_rules import UAROETCohortState, _profile
from .uaroet_desktop_study import UAROETDesktopScenario, UAROETDesktopStudy


@dataclass(frozen=True)
class UAROETCalendarLook:
    """Small audit record; full posterior draws are available only in the callback."""

    clock: float
    enrolled: int
    action: str
    selected: int | None
    reason: str
    final: bool
    status: tuple[int, ...]
    draws_per_chain: int
    work_evaluations: int


@dataclass(frozen=True)
class UAROETCalendarTrial:
    scenario: int
    profile: str
    selected: int | None
    stopped_early: bool
    patients: tuple[UAROETTimedPatient, ...]
    looks: tuple[UAROETCalendarLook, ...]
    followup_duration: float
    decision_time: float
    discarded_arrivals: int


class UAROETSimulationHold(RuntimeError):
    """An unresolved computation must not enter clinical operating characteristics."""

    def __init__(self, clock: float, decision: UAROETDesktopDecision):
        super().__init__(f"UAROET computation held at time {clock}: {decision.reason}")
        self.clock = clock
        self.decision = decision


def run_uaroet_desktop_trial(
    study: UAROETDesktopStudy,
    scenario: UAROETDesktopScenario,
    *,
    rng: np.random.Generator,
    profile: str = "safe",
    target: float | None = None,
    chunk_draws: int = 1000,
    max_draws: int = 100_000,
    max_evaluations: int = 2_000_000,
    maximum_rhat: float = 1.05,
    max_discarded_arrivals: int = 100_000,
    on_look: Callable[[float, UAROETDesktopDecision], None] | None = None,
) -> UAROETCalendarTrial:
    """Run one trial, retaining all generated outcomes and compact decision records.

    First arrival is zero, subsequent intervals are exponential. Every decision
    uses observed data at that clock and live cohort state. Computational holds
    raise; they never become clinical stops. Early clinical stopping ends accrual
    without a final refit, but outcomes are followed for allocation summaries.

    Legacy mode rounds outcome delays and preserves native suspension: only
    closing an already open new-dose cohort triggers waiting, only its last
    patient's completion sets the deadline, and intervening arrivals are lost.
    The safe profile does not define this optional suspension policy and rejects
    it. RNG sequence identity with the native executable is not claimed.
    """
    _profile(profile)
    if not isinstance(study, UAROETDesktopStudy) or study.patients:
        raise ValueError("simulation requires a study without patient rows")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("require an explicit NumPy Generator")
    if not isinstance(scenario, UAROETDesktopScenario) or (
        scenario.efficacy.shape != (study.dose_count, study.efficacy_levels)
        or scenario.toxicity.shape != (study.dose_count, study.toxicity_levels)
    ):
        raise ValueError("scenario must match the study grid")
    p, settings = study.trial_parameters, study.simulation_parameters
    suspend = bool(p.get("SuspendAccrualForFullEvalOfNewDoseCohort", 0))
    if profile == "safe" and (suspend or not p.get("CheckAcceptabilityWithinCohort", 1)):
        raise ValueError("suspension or within-cohort bypass requires the legacy profile")
    if on_look is not None and not callable(on_look):
        raise ValueError("on_look must be callable")
    discarded_limit = _integer(max_discarded_arrivals, "max_discarded_arrivals", 0, 1_000_000)
    # Validate inference controls before generating any patients or consuming RNG.
    if target is None:
        if "TargetMCSERatio_Sim" in settings:
            raise ValueError("supply target explicitly when TargetMCSERatio_Sim is present")
        target = settings.get("TargetMCSERatio", 0.03)
    ratio = _real_matrix(target, "target")
    rhat = _real_matrix(maximum_rhat, "maximum_rhat")
    if ratio.ndim or not 0 < float(ratio) < 1 or rhat.ndim or float(rhat) < 1:
        raise ValueError("target must be in(0,1) and maximum_rhat at least one")
    chunk = _integer(chunk_draws, "chunk_draws", 8, 100_000)
    limit = _integer(max_draws, "max_draws", 8, 1_000_000)
    _integer(max_evaluations, "max_evaluations", 1, 128_000_000)
    if limit < max(8, int(settings.get("BurnInSamples", 1000))) or chunk > limit:
        raise ValueError("max_draws must cover burn-in length and chunk_draws")
    if int(settings.get("NumChains", 4)) * limit * study.dose_count > 20_000_000:
        raise ValueError("retained utility budget exceeds twenty million values")
    if chunk * study.dose_count * study.efficacy_levels * study.toxicity_levels > 2_000_000:
        raise ValueError("chunk joint posterior exceeds two million cells")
    joint = scenario.joint()
    cdf = np.cumsum(joint.reshape(study.dose_count, -1), axis=1)
    cdf[:, -1] = 1.0
    state = UAROETCohortState((0,) * study.dose_count)
    patients: list[UAROETTimedPatient] = []
    looks: list[UAROETCalendarLook] = []
    clock = deadline = 0.0
    discarded = 0
    selected = None
    stopped = False
    while True:
        final = len(patients) == int(p["NumPatients"])
        if final:
            followup = max(patient.completion_time for patient in patients)
            clock = followup + 1.0 if profile == "legacy_1_8" else max(clock, followup)
        observed = uaroet_study_as_of(study, tuple(patients), clock, profile=profile)
        decision = recommend_uaroet_desktop(
            observed,
            rng=rng,
            profile=profile,
            final=final,
            cohort_state=state,
            target=float(ratio),
            chunk_draws=chunk,
            max_draws=limit,
            max_evaluations=max_evaluations,
            maximum_rhat=float(rhat),
            conduct=False,
        )
        if on_look is not None:
            on_look(clock, decision)
        if decision.action == "hold":
            raise UAROETSimulationHold(clock, decision)
        fit = decision.posterior
        looks.append(
            UAROETCalendarLook(
                clock,
                len(patients),
                decision.action,
                decision.selected,
                decision.reason,
                final,
                decision.status,
                0 if fit is None else fit.draws_per_chain,
                0 if fit is None else fit.work_evaluations,
            )
        )
        if final or decision.action == "stop":
            selected = decision.selected if final else None
            stopped = not final
            break
        dose = decision.selected
        if decision.action != "treat" or dose is None:
            raise RuntimeError("unexpected assignment action")
        cell = int(np.searchsorted(cdf[dose], rng.random(), side="right"))
        efficacy, toxicity = divmod(cell, study.toxicity_levels)
        e_delay, t_delay = uaroet_outcome_delays(
            [p["MinDaysToEfficacy"], p["MaxDaysToEfficacy"]],
            [p["MinDaysToToxicity"], p["MaxDaysToToxicity"]],
            rng.random(2),
            profile=profile,
        )
        patient = UAROETTimedPatient(
            len(patients) + 1, dose, efficacy, toxicity, clock, e_delay, t_delay
        )
        patients.append(patient)
        previous = state
        state = state.advance(
            dose,
            new_size=int(p["CohortSize_NewDose"]),
            old_size=int(p["CohortSize_OldDose"]),
            profile=profile,
        )
        if suspend and previous.new_open and not state.new_open:
            deadline = max(deadline, patient.completion_time)
        if len(patients) < int(p["NumPatients"]):
            clock, lost = _advance_arrival(
                clock,
                deadline,
                lambda: rng.exponential(p["MeanDaysBetweenArrivals"]),
                discarded_limit - discarded,
            )
            discarded += lost
    return UAROETCalendarTrial(
        scenario.identifier,
        profile,
        selected,
        stopped,
        tuple(patients),
        tuple(looks),
        max((patient.completion_time for patient in patients), default=0.0),
        clock,
        discarded,
    )
