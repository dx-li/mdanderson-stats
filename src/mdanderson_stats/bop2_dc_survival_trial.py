"""Calendar replay and bounded operating characteristics for BOP2-DC survival."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, scalar
from .bop2_dc_survival import BOP2DCSurvivalDesign, BOP2DCSurvivalState
from .bop2_survival_trial import _survival_paths, _trial_count

_DECISIONS = ("stop_no_go", "final_go", "final_consider", "final_no_go")
_MAX_SIMULATION_CELLS = 1_000_000
_MAX_LOOK_WORK = 30_000_000
_MAX_RETAINED_CELLS = 2_000_000


@dataclass(frozen=True)
class BOP2DCSurvivalTrial:
    """One survival replay, retaining a compact state at each analysis."""

    calendar_times: FloatArray
    states: tuple[BOP2DCSurvivalState, ...]
    enrolled: int
    events: int
    total_time: float
    duration: float
    decision: str


@dataclass(frozen=True)
class BOP2DCSurvivalSimulation:
    """Monte Carlo operating characteristics plus per-trial audit summaries."""

    trials: int
    decision_labels: tuple[str, ...]
    decision_count: NDArray[np.int64]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    mean_enrollment: float
    enrollment_mcse: float
    mean_events: float
    events_mcse: float
    mean_total_time: float
    total_time_mcse: float
    mean_duration: float
    duration_mcse: float
    sample_size: NDArray[np.int64]
    events: NDArray[np.int64]
    total_time: FloatArray
    duration: FloatArray
    decision: NDArray[np.str_]
    analysis_time: FloatArray
    rng_seed: int


def run_bop2_dc_survival_trial(
    design: BOP2DCSurvivalDesign,
    enrollment_times: ArrayLike,
    event_times: ArrayLike,
    *,
    final_followup: float,
) -> BOP2DCSurvivalTrial:
    """Replay accrued patients with administrative censoring at each scheduled look.

    ``event_times`` are durations from enrollment; positive infinity denotes no event.
    Interim analyses occur at the enrollment time of the last patient in each look.
    The final analysis occurs after ``final_followup`` beyond the last enrollment.
    Events at the analysis boundary count as observed. Input order breaks tied enrollment
    times. The replay stops at the first BOP2-DC no-go decision.
    """
    _check_design(design)
    _check_vector_shape(enrollment_times, design.max_subjects, "enrollment_times")
    _check_vector_shape(event_times, design.max_subjects, "event_times")
    enrolled = _real_array(enrollment_times, "enrollment_times")
    durations = _real_array(event_times, "event_times")
    followup = scalar(final_followup, "final_followup")
    if np.any(enrolled < 0) or np.any(np.diff(enrolled) < 0):
        raise ValueError("enrollment_times must be ordered and nonnegative")
    if np.any(np.isnan(durations) | (durations < 0)):
        raise ValueError("event_times must be nonnegative or positive infinity")
    if followup < 0:
        raise ValueError("final_followup must be nonnegative")
    final_clock = float(enrolled[-1] + followup)
    if not np.isfinite(final_clock):
        raise ArithmeticError("final analysis calendar time overflows")
    if followup > 0 and final_clock <= enrolled[-1]:
        raise ArithmeticError("positive final_followup is lost at this calendar-time scale")

    states: list[BOP2DCSurvivalState] = []
    clocks: list[float] = []
    terminal_summary: tuple[int, int, float] | None = None
    for raw_n in design.looks:
        n = int(raw_n)
        is_final = n == design.max_subjects
        last_enrollment = float(enrolled[n - 1])
        clock = float(last_enrollment + (followup if is_final else 0.0))
        elapsed = last_enrollment - enrolled[:n]
        if is_final:
            elapsed = elapsed + followup
        if np.any(elapsed < 0) or np.any(~np.isfinite(elapsed)):
            raise ArithmeticError("as-of follow-up is not representable")
        observed = np.minimum(durations[:n], elapsed)
        events = int(np.count_nonzero(durations[:n] <= elapsed))
        total_time = float(np.sum(observed, dtype=np.float64))
        if not np.isfinite(total_time):
            raise ArithmeticError("total observed time overflows")
        state = design.monitor(n, events, total_time)
        states.append(state)
        clocks.append(clock)
        terminal_summary = (n, events, total_time)
        if str(state.decision.item()) != "continue":
            break
    assert terminal_summary is not None
    n, events, total_time = terminal_summary
    final_state = str(states[-1].decision.item())
    return BOP2DCSurvivalTrial(
        _freeze(np.asarray(clocks, dtype=np.float64)),
        tuple(states),
        n,
        events,
        total_time,
        clocks[-1],
        final_state,
    )


def simulate_bop2_dc_survival(
    design: BOP2DCSurvivalDesign,
    true_median: float,
    *,
    accrual_rate: float,
    final_followup: float,
    n_trials: int = 10_000,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
) -> BOP2DCSurvivalSimulation:
    """Simulate exponential survival trials and summarize BOP2-DC decisions.

    Event times are exponential with mean ``true_median / log(2)``. Administrative
    censoring is used. ``arrival='fixed'`` or ``'poisson'`` selects the existing Python
    accrual schedule; the paper does not specify an arrival-gap distribution. ``rng_seed``
    in the result replays the entire aggregate call with the same arguments.

    The simulation is serial and bounded by path-cell, repeated-look work, and retained
    output limits. No native RNG or timing-schedule parity is claimed.
    """
    _check_design(design)
    trials = _trial_count(n_trials)
    median, rate, followup = (
        scalar(true_median, "true_median"),
        scalar(accrual_rate, "accrual_rate"),
        scalar(final_followup, "final_followup"),
    )
    if median <= 0 or rate <= 0 or followup < 0:
        raise ValueError("require positive true_median/accrual_rate and nonnegative follow-up")
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be 'fixed' or 'poisson'")
    n = design.max_subjects
    path_cells = trials * n
    look_work = trials * int(np.sum(design.looks, dtype=np.int64))
    retained_cells = trials * 6 + len(_DECISIONS) * 3
    if path_cells > _MAX_SIMULATION_CELLS:
        raise ValueError("simulation exceeds the survival path cell budget")
    if look_work > _MAX_LOOK_WORK:
        raise ValueError("simulation exceeds the repeated-look work budget")
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("simulation exceeds the retained result cell budget")
    if not np.isfinite(median / np.log(2.0)) or not np.isfinite(1.0 / rate):
        raise ArithmeticError("event or accrual time scale is not representable")
    if median / np.log(2.0) == 0 or 1.0 / rate == 0:
        raise ArithmeticError("event or accrual time scale underflows")

    seed = _replay_seed(rng)
    generator = np.random.default_rng(seed)
    sample_size = np.empty(trials, dtype=np.int64)
    events = np.empty(trials, dtype=np.int64)
    total_time = np.empty(trials, dtype=np.float64)
    duration = np.empty(trials, dtype=np.float64)
    analysis_time = np.empty(trials, dtype=np.float64)
    decision = np.empty(trials, dtype="U16")

    # The shared generator and chunked path helper keep memory independent of trial count.
    for start, enrolled, event_times, fup in _survival_paths(
        n, median, rate, followup, trials, arrival, generator
    ):
        batch = enrolled.shape[0]
        active = np.ones(batch, dtype=bool)
        for raw_n in design.looks:
            look = int(raw_n)
            is_final = look == n
            last = enrolled[:, look - 1]
            clock = last + (fup if is_final else 0.0)
            if np.any(~np.isfinite(clock)) or (is_final and fup > 0 and np.any(clock <= last)):
                raise ArithmeticError("analysis calendar time is not representable")
            elapsed = last[:, None] - enrolled[:, :look]
            if is_final:
                elapsed = elapsed + fup
            observed = np.minimum(event_times[:, :look], elapsed)
            event_count = np.count_nonzero(event_times[:, :look] <= elapsed, axis=-1)
            exposure = np.sum(observed, axis=-1, dtype=np.float64)
            if np.any(~np.isfinite(exposure)):
                raise ArithmeticError("simulated total observed time overflows")
            state = design.monitor(look, event_count, exposure)
            decisions = state.decision.astype("U16")
            ended = active & (decisions != "continue")
            indices = start + np.flatnonzero(ended)
            sample_size[indices] = look
            events[indices] = event_count[ended]
            total_time[indices] = exposure[ended]
            duration[indices] = clock[ended]
            analysis_time[indices] = clock[ended]
            decision[indices] = decisions[ended]
            active[ended] = False

    labels = _DECISIONS
    counts = np.asarray([np.count_nonzero(decision == label) for label in labels], dtype=np.int64)
    probabilities = counts.astype(np.float64) / trials
    mcse = np.sqrt(probabilities * (1.0 - probabilities) / trials)
    return BOP2DCSurvivalSimulation(
        trials,
        labels,
        _freeze(counts),
        _freeze(probabilities),
        _freeze(mcse),
        _mean(sample_size.astype(np.float64)),
        _sample_mcse(sample_size.astype(np.float64)),
        _mean(events.astype(np.float64)),
        _sample_mcse(events.astype(np.float64)),
        _mean(total_time),
        _sample_mcse(total_time),
        _mean(duration),
        _sample_mcse(duration),
        _freeze(sample_size),
        _freeze(events),
        _freeze(total_time),
        _freeze(duration),
        _freeze(decision),
        _freeze(analysis_time),
        seed,
    )


def _check_design(design: BOP2DCSurvivalDesign) -> None:
    if not isinstance(design, BOP2DCSurvivalDesign):
        raise ValueError("design must be a BOP2DCSurvivalDesign")


def _check_vector_shape(value: ArrayLike, length: int, name: str) -> None:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"{name} must have length {length}")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")


def _real_array(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.asarray(value, dtype=np.float64)
    if np.any(np.isnan(result)) or (name == "enrollment_times" and np.any(~np.isfinite(result))):
        raise ValueError(f"{name} contains invalid non-finite values")
    return result


def _replay_seed(rng: np.random.Generator | int | None) -> int:
    if isinstance(rng, np.random.Generator):
        source = rng
    elif rng is not None:
        if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)):
            raise ValueError("rng must be a nonnegative seed or NumPy Generator")
        seed = int(rng)
        if not 0 <= seed < 2**64:
            raise ValueError("integer rng seed must lie in [0, 2**64)")
        return seed
    else:
        source = np.random.default_rng()
    return int(source.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def _sample_mcse(values: FloatArray) -> float:
    if values.size < 2:
        return float("nan")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    scaled = values / scale
    error = scale * float(np.std(scaled, ddof=1) / np.sqrt(values.size))
    if not np.isfinite(error):
        raise ArithmeticError("Monte Carlo standard error is not representable")
    return error


def _mean(values: FloatArray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    result = scale * float(np.mean(values / scale))
    if not np.isfinite(result):
        raise ArithmeticError("Monte Carlo mean is not representable")
    return result


def _freeze(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result
