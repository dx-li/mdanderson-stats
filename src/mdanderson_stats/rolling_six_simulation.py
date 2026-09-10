"""Rolling Six calendar simulations with shared calibrated event generation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._tite_simulation import CalendarSimulation, simulate_calendar
from ._validation import FloatArray, finite
from .rolling_six import RollingSixDesign
from .rolling_six_trial import RollingSixTrial, run_rolling_six_trial


@dataclass(frozen=True)
class RollingSixSimulation(CalendarSimulation):
    """Selection probabilities include the highest planned dose when it is recommended."""

    selection_status: tuple[str, ...]


def simulate_rolling_six(
    design: RollingSixDesign,
    true_toxicity: ArrayLike,
    window: float,
    accrual_rate: float,
    *,
    max_patients: int | None = None,
    trials: int = 1000,
    start_dose: int = 1,
    arrival: str = "fixed",
    event_distribution: str = "uniform",
    late_probability: ArrayLike | None = None,
    event_trimester_probabilities: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> RollingSixSimulation:
    """Simulate up to six patients per dose, subject to an overall cap of 200.

    The default planned cap is min(6*doses,200). Selection at the highest dose
    means a dose recommendation; it does not establish an upper toxicity limit.
    """
    p = finite(true_toxicity, "true_toxicity")
    cap = min(6 * p.size, 200) if max_patients is None else max_patients

    def replay(
        gaps: FloatArray, delays: FloatArray, duration: float, size: int, start: int
    ) -> RollingSixTrial:
        return run_rolling_six_trial(design, gaps, delays, duration, start_dose=start)

    result = simulate_calendar(
        replay,
        p,
        window,
        accrual_rate,
        cap,
        1,
        trials,
        start_dose,
        arrival,
        event_distribution,
        late_probability,
        event_trimester_probabilities,
        rng,
    )
    return RollingSixSimulation(
        result.patients,
        result.toxicities,
        result.selected_dose,
        result.selection_probability,
        result.selection_mcse,
        result.duration,
        result.suspension_time,
        result.stop_reason,
        tuple(design.select_mtd(n, y).status for n, y in zip(result.patients, result.toxicities)),
    )
