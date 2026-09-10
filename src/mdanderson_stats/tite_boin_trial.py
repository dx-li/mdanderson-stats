"""Calendar-time TITE-BOIN trial replay."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ._tite_calendar import CalendarStep, CalendarTrial, run_calendar_trial
from ._validation import FloatArray, scalar
from .boin import BOINDesign
from .tite_boin import TITEBOINDecision, tite_boin_decision
from .tite_keyboard import toxicity_followup_weights


@dataclass(frozen=True)
class TITEBOINStep(CalendarStep[TITEBOINDecision]):
    pass


@dataclass(frozen=True)
class TITEBOINTrial(CalendarTrial[TITEBOINStep]):
    pass


def run_tite_boin_trial(
    design: BOINDesign,
    interarrival: ArrayLike,
    dlt_delays: ArrayLike,
    window: float,
    *,
    cohort_size: int = 3,
    start_dose: int = 1,
    trimester_probabilities: ArrayLike | None = None,
    minimum_complete_fraction: float = 0.51,
    minimum_pending_followup: float = 0.25,
) -> TITEBOINTrial:
    """Conduct one trial from potential outcomes without exposing future events.

    dlt_delays[patient,dose] is time since enrollment to DLT, or +inf for no DLT
    within the window. Arrival gaps restart after suspensions; no queue builds up.
    Cohorts have a fixed dose and staggered enrollment. Decisions occur before
    each new cohort, at outcome ascertainments while waiting, and when the
    minimum pending-follow-up threshold is reached.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    toxicity_followup_weights([], window, trimester_probabilities=trimester_probabilities)
    complete = scalar(minimum_complete_fraction, "minimum_complete_fraction")
    minimum = scalar(minimum_pending_followup, "minimum_pending_followup")
    if not 0.25 <= complete <= 1 or not 0 <= minimum <= 1:
        raise ValueError("require completion fraction in [.25,1] and minimum follow-up in [0,1]")

    def decide(
        n: ArrayLike, y: ArrayLike, times: list[FloatArray], current: int, excluded: ArrayLike
    ) -> TITEBOINDecision:
        return tite_boin_decision(
            design,
            n,
            y,
            times,
            current,
            window,
            trimester_probabilities=trimester_probabilities,
            minimum_complete_fraction=complete,
            minimum_pending_followup=minimum,
            eliminated=excluded,
        )

    result = run_calendar_trial(
        design,
        interarrival,
        dlt_delays,
        window,
        cohort_size,
        start_dose,
        decide,
        TITEBOINStep,
        minimum,
    )
    return TITEBOINTrial(
        result.enrollment_times,
        result.assigned_doses,
        result.dlt_times,
        result.patients,
        result.toxicities,
        result.selected_dose,
        result.eliminated,
        result.stop_reason,
        result.final_time,
        result.suspension_time,
        result.steps,
    )
