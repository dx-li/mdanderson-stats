"""Calendar-time TITE-BOIN simulations with calibrated DLT timing scenarios."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._tite_simulation import CalendarSimulation, simulate_calendar
from ._validation import FloatArray
from .boin import BOINDesign
from .tite_boin_trial import TITEBOINTrial, run_tite_boin_trial


@dataclass(frozen=True)
class TITEBOINSimulation(CalendarSimulation):
    pass


def simulate_tite_boin(
    design: BOINDesign,
    true_toxicity: ArrayLike,
    window: float,
    accrual_rate: float,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    arrival: str = "fixed",
    event_distribution: str = "uniform",
    late_probability: ArrayLike | None = None,
    event_trimester_probabilities: ArrayLike | None = None,
    trimester_probabilities: ArrayLike | None = None,
    minimum_complete_fraction: float = 0.51,
    minimum_pending_followup: float = 0.25,
    rng: int | np.random.Generator | None = None,
) -> TITEBOINSimulation:
    """Simulate binary DLT incidence and conditional DLT time separately.

    accrual_rate is patients per time unit used for window. The first arrival
    also has an interarrival gap. 'fixed' uses gaps 1/rate; 'exponential' draws
    independent exponential gaps. True event timing and analysis weights are
    separate: both default to uniform but can have distinct trimester masses.
    Weibull/log-logistic scenarios use late_probability to calibrate the fraction
    of DLTs in the late half of the window, independently of analysis weights.
    """

    def replay(
        gaps: FloatArray, delays: FloatArray, duration: float, size: int, start: int
    ) -> TITEBOINTrial:
        return run_tite_boin_trial(
            design,
            gaps,
            delays,
            duration,
            cohort_size=size,
            start_dose=start,
            trimester_probabilities=trimester_probabilities,
            minimum_complete_fraction=minimum_complete_fraction,
            minimum_pending_followup=minimum_pending_followup,
        )

    result = simulate_calendar(
        replay,
        true_toxicity,
        window,
        accrual_rate,
        cohorts,
        cohort_size,
        trials,
        start_dose,
        arrival,
        event_distribution,
        late_probability,
        event_trimester_probabilities,
        rng,
    )
    return TITEBOINSimulation(
        result.patients,
        result.toxicities,
        result.selected_dose,
        result.selection_probability,
        result.selection_mcse,
        result.duration,
        result.suspension_time,
        result.stop_reason,
    )
