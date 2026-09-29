"""Bounded operating characteristics for randomized BOP2-DC survival monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._validation import FloatArray, scalar
from .bop2_dc_randomized_survival import (
    _QUAD_EVALUATIONS_PER_INTEGRAL,
    BOP2DCRandomizedSurvivalDesign,
    run_bop2_dc_randomized_survival_trial,
)
from .bop2_dc_survival_trial import (
    _freeze,
    _mean,
    _replay_seed,
    _sample_mcse,
    _trial_count,
)

_DECISIONS = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_MAX_SIMULATION_CELLS = 1_000_000
_MAX_LOOK_WORK = 30_000_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_TOTAL_QUADRATURE_EVALUATIONS = 100_000_000
_LOG2 = np.log(2.0)


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalSimulation:
    """Monte Carlo decisions and compact per-trial arm-stratified summaries."""

    trials: int
    decision_labels: tuple[str, ...]
    decision_count: NDArray[np.int64]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    mean_enrollment: float
    enrollment_mcse: float
    mean_events: FloatArray
    events_mcse: FloatArray
    mean_exposure: FloatArray
    exposure_mcse: FloatArray
    mean_duration: float
    duration_mcse: float
    enrolled: NDArray[np.int64]
    arm_n: NDArray[np.int64]
    arm_events: NDArray[np.int64]
    arm_exposure: FloatArray
    duration: FloatArray
    decision: NDArray[np.str_]
    analysis_time: FloatArray
    rng_seed: int


def simulate_bop2_dc_randomized_survival(
    design: BOP2DCRandomizedSurvivalDesign,
    control_true_median: float,
    treatment_true_median: float,
    *,
    accrual_rate: float,
    final_followup: float,
    n_trials: int = 100,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
) -> BOP2DCRandomizedSurvivalSimulation:
    """Simulate fixed-allocation trials with arm-specific exponential event times.

    The design's 0/1 assignment tape is followed exactly. Event times are exponential
    with means ``true_median / log(2)``. ``arrival='fixed'`` enrolls at integer
    multiples of ``1/accrual_rate``; ``'poisson'`` uses exponential gaps with that
    mean. These are Python scheduling conventions, not native RNG or calendar parity.

    A single integer ``rng_seed`` makes the aggregate call replayable with identical
    arguments. The runner executes trials serially and retains only scalar summaries,
    not each trial's posterior states. The default 100 trials is a workload choice:
    quadrature cost can be substantial when both margins are nonzero.
    """
    if not isinstance(design, BOP2DCRandomizedSurvivalDesign):
        raise ValueError("design must be a BOP2DCRandomizedSurvivalDesign")
    trials = _trial_count(n_trials)
    control_median = scalar(control_true_median, "control_true_median")
    treatment_median = scalar(treatment_true_median, "treatment_true_median")
    rate = scalar(accrual_rate, "accrual_rate")
    followup = scalar(final_followup, "final_followup")
    if control_median <= 0 or treatment_median <= 0 or rate <= 0 or followup < 0:
        raise ValueError("require positive arm medians/accrual_rate and nonnegative final_followup")
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be 'fixed' or 'poisson'")

    n = design.max_subjects
    interval = 1.0 / rate
    event_means = np.asarray([control_median, treatment_median], dtype=np.float64) / _LOG2
    if (
        not np.isfinite(interval)
        or interval <= 0
        or np.any(~np.isfinite(event_means))
        or np.any(event_means <= 0)
    ):
        raise ArithmeticError("event or accrual time scale is not representable")
    expected_last = n * interval
    expected_final = expected_last + followup
    if not np.isfinite(expected_last) or not np.isfinite(expected_final):
        raise ArithmeticError("planned accrual calendar time overflows")
    if followup > 0 and expected_final <= expected_last:
        raise ArithmeticError("positive final_followup is lost at this calendar-time scale")

    path_cells = trials * n
    look_work = trials * int(np.sum(design.looks, dtype=np.int64))
    per_trial_live_cells = 2 * 17 + 3 + 2 * n + 18 * len(design.looks)
    retained_cells = trials * per_trial_live_cells + len(_DECISIONS) * 6 + 16
    margin_count = int(design.median_lrv != 0) + int(design.median_cmv != 0)
    quadrature_work = trials * len(design.looks) * margin_count * _QUAD_EVALUATIONS_PER_INTEGRAL
    if path_cells > _MAX_SIMULATION_CELLS:
        raise ValueError("simulation exceeds the survival path cell budget")
    if look_work > _MAX_LOOK_WORK:
        raise ValueError("simulation exceeds the repeated-look work budget")
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("simulation exceeds the retained result cell budget")
    if quadrature_work > _MAX_TOTAL_QUADRATURE_EVALUATIONS:
        raise ValueError("simulation exceeds the total inverse-gamma quadrature work budget")

    seed = _replay_seed(rng)
    generator = np.random.default_rng(seed)
    enrollment = np.empty(trials, dtype=np.int64)
    arm_n = np.empty((trials, 2), dtype=np.int64)
    arm_events = np.empty((trials, 2), dtype=np.int64)
    arm_exposure = np.empty((trials, 2), dtype=np.float64)
    duration = np.empty(trials, dtype=np.float64)
    analysis_time = np.empty(trials, dtype=np.float64)
    decision = np.empty(trials, dtype="U16")
    assignments = design.arm_assignments

    for trial_index in range(trials):
        if arrival == "fixed":
            enrolled_at = np.arange(1, n + 1, dtype=np.float64) * interval
        else:
            enrolled_at = np.cumsum(generator.exponential(interval, size=n), dtype=np.float64)
        event_times = generator.exponential(scale=event_means[assignments], size=n)
        if np.any(~np.isfinite(enrolled_at)) or np.any(~np.isfinite(event_times)):
            raise ArithmeticError("simulated calendar or event times overflow")

        replay = run_bop2_dc_randomized_survival_trial(
            design,
            enrolled_at,
            event_times,
            final_followup=followup,
        )
        state = replay.states[-1]
        enrollment[trial_index] = replay.enrolled
        arm_n[trial_index] = [state.control_n.item(), state.treatment_n.item()]
        arm_events[trial_index] = [state.control_events.item(), state.treatment_events.item()]
        arm_exposure[trial_index] = [
            state.control_exposure.item(),
            state.treatment_exposure.item(),
        ]
        duration[trial_index] = replay.calendar_times[-1]
        analysis_time[trial_index] = replay.calendar_times[-1]
        decision[trial_index] = replay.decision

    decision_count = np.asarray(
        [np.count_nonzero(decision == label) for label in _DECISIONS], dtype=np.int64
    )
    if int(np.sum(decision_count, dtype=np.int64)) != trials:
        raise ArithmeticError("simulated terminal decisions do not conserve trial count")
    decision_probability = decision_count.astype(np.float64) / trials
    decision_mcse = np.sqrt(decision_probability * (1.0 - decision_probability) / trials)
    events_float = arm_events.astype(np.float64)
    return BOP2DCRandomizedSurvivalSimulation(
        trials,
        _DECISIONS,
        _freeze(decision_count),
        _freeze(decision_probability),
        _freeze(decision_mcse),
        _mean(enrollment.astype(np.float64)),
        _sample_mcse(enrollment.astype(np.float64)),
        _freeze(np.asarray([_mean(events_float[:, arm]) for arm in range(2)])),
        _freeze(np.asarray([_sample_mcse(events_float[:, arm]) for arm in range(2)])),
        _freeze(np.asarray([_mean(arm_exposure[:, arm]) for arm in range(2)])),
        _freeze(np.asarray([_sample_mcse(arm_exposure[:, arm]) for arm in range(2)])),
        _mean(duration),
        _sample_mcse(duration),
        _freeze(enrollment),
        _freeze(arm_n),
        _freeze(arm_events),
        _freeze(arm_exposure),
        _freeze(duration),
        _freeze(decision),
        _freeze(analysis_time),
        seed,
    )
