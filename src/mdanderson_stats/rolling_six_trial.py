"""Patient-by-patient Rolling Six replay on the shared calendar scheduler."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._tite_calendar import CalendarStep, CalendarTrial, run_calendar_trial
from ._validation import FloatArray
from .rolling_six import RollingSixDecision, RollingSixDesign


@dataclass(frozen=True)
class RollingSixStep(CalendarStep[RollingSixDecision]):
    pass


@dataclass(frozen=True)
class RollingSixTrial(CalendarTrial[RollingSixStep]):
    selection_status: str


def run_rolling_six_trial(
    design: RollingSixDesign,
    interarrival: ArrayLike,
    dlt_delays: ArrayLike,
    window: float,
    *,
    start_dose: int = 1,
) -> RollingSixTrial:
    """Enroll one patient per decision, with no queue during suspensions.

    Potential outcomes and arrival gaps use the same conventions as TITE-BOIN.
    Final follow-up occurs even when enrollment ends at its supplied patient cap.
    """
    if not isinstance(design, RollingSixDesign):
        raise ValueError("design must be a RollingSixDesign")

    def decide(
        n: ArrayLike, y: ArrayLike, times: list[FloatArray], current: int, excluded: ArrayLike
    ) -> RollingSixDecision:
        return design.next_dose(
            n, y, np.array([len(t) for t in times]), current, eliminated=excluded
        )

    result = run_calendar_trial(
        design, interarrival, dlt_delays, window, 1, start_dose, decide, RollingSixStep
    )
    selected = design.select_mtd(result.patients, result.toxicities, eliminated=result.eliminated)
    return RollingSixTrial(
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
        selected.status,
    )
