"""Calendar replay and Monte Carlo OC for single-arm BOP2 survival monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer, _owned
from .bop2_survival import BOP2SurvivalDesign, BOP2SurvivalState


@dataclass(frozen=True)
class BOP2SurvivalTrial:
    calendar_times: FloatArray
    states: tuple[BOP2SurvivalState, ...]


@dataclass(frozen=True)
class BOP2SurvivalSimulation:
    sample_size: FloatArray
    events: FloatArray
    calendar_time: FloatArray
    success: FloatArray
    success_probability: float
    success_mcse: float
    expected_sample_size: float


def run_bop2_survival_trial(
    design: BOP2SurvivalDesign,
    enrollment_times: ArrayLike,
    event_times: ArrayLike,
    *,
    final_followup: float,
) -> BOP2SurvivalTrial:
    """Interims at enrollment counts; final analysis after the last enrollment plus follow-up.

    Event times are durations from enrollment, with +inf for no event. Administrative
    censoring applies at each analysis. Replay ends at the first stopping decision.
    """
    enrolled = finite(enrollment_times, "enrollment_times")
    durations = np.asarray(event_times, dtype=float)
    followup = scalar(final_followup, "final_followup")
    if (
        enrolled.shape != (design.max_subjects,)
        or durations.shape != enrolled.shape
        or np.any(enrolled < 0)
        or np.any(np.diff(enrolled) < 0)
    ):
        raise ValueError("require N ordered nonnegative enrollment times and N event durations")
    if np.any(np.isnan(durations) | (durations < 0)) or followup < 0:
        raise ValueError("event durations must be nonnegative or +inf, final_followup nonnegative")
    states = []
    clocks = []
    for n in design.looks:
        clock = float(enrolled[n - 1] + (followup if n == design.max_subjects else 0))
        if not np.isfinite(clock):
            raise ArithmeticError("analysis calendar time overflows")
        observed = clock - enrolled[:n]
        state = design.monitor(
            int(np.count_nonzero(durations[:n] <= observed)),
            float(np.minimum(durations[:n], observed).sum()),
            int(n),
        )
        states.append(state)
        clocks.append(clock)
        if state.decision != "continue":
            break
    return BOP2SurvivalTrial(_owned(clocks), tuple(states))


def simulate_bop2_survival(
    design: BOP2SurvivalDesign,
    true_median: float,
    *,
    accrual_rate: float,
    final_followup: float,
    n_trials: int = 10000,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
) -> BOP2SurvivalSimulation:
    """Vectorized exponential-event simulation with fixed or Poisson enrollment.

    Includes one interarrival interval before the first enrollment. Uses administrative
    censoring only. Operating-characteristic estimates have Monte Carlo uncertainty.
    """
    median, rate, fup = (
        scalar(true_median, "true_median"),
        scalar(accrual_rate, "accrual_rate"),
        scalar(final_followup, "final_followup"),
    )
    trials = _integer(n_trials, "n_trials")
    if median <= 0 or rate <= 0 or fup < 0 or not 1 <= trials <= 100000:
        raise ValueError(
            "require positive median/rate, nonnegative follow-up, n_trials in [1,100000]"
        )
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be fixed or poisson")
    mean = median / np.log(2)
    interval = 1 / rate
    if not np.isfinite(mean) or not np.isfinite(interval):
        raise ArithmeticError("event or accrual time scale is not representable")
    generator = np.random.default_rng(rng)
    ns = np.empty(trials)
    ds = np.empty(trials)
    clocks = np.empty(trials)
    success = np.empty(trials)
    for start in range(0, trials, 5000):
        stop = min(start + 5000, trials)
        size = stop - start
        enrolled = (
            np.broadcast_to(
                np.arange(1, design.max_subjects + 1) * interval, (size, design.max_subjects)
            )
            if arrival == "fixed"
            else generator.exponential(interval, (size, design.max_subjects)).cumsum(axis=-1)
        )
        times = generator.exponential(mean, (size, design.max_subjects))
        if np.any(~np.isfinite(enrolled)) or np.any(~np.isfinite(times)):
            raise ArithmeticError("simulated calendar or event times overflow")
        active = np.ones(size, dtype=bool)
        for n in design.looks:
            clock = enrolled[:, n - 1] + (fup if n == design.max_subjects else 0)
            if np.any(~np.isfinite(clock)):
                raise ArithmeticError("analysis calendar time overflows")
            observed = clock[:, None] - enrolled[:, :n]
            events = np.count_nonzero(times[:, :n] <= observed, axis=-1)
            state = design.monitor(events, np.minimum(times[:, :n], observed).sum(axis=-1), int(n))
            ended = active & (state.decision != "continue")
            indices = start + np.flatnonzero(ended)
            ns[indices] = n
            ds[indices] = events[ended]
            clocks[indices] = clock[ended]
            success[indices] = state.decision[ended] == "final_positive"
            active[ended] = False
    probability = float(success.mean())
    return BOP2SurvivalSimulation(
        _owned(ns),
        _owned(ds),
        _owned(clocks),
        _owned(success),
        probability,
        float(np.sqrt(probability * (1 - probability) / trials)),
        float(ns.mean()),
    )
