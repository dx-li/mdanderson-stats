"""Scheduled calendar replay and operating-characteristic simulation for Phase2Delay.

Calendar looks are an explicit discrete protocol. Accrual and follow-up continue
between looks; the trial enrolls patients as they arrive and analyzes only at the
supplied decision times.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .phase2delay import _integer, phase2_delay_monitor

FinalAnalysis = Literal["at_enrollment_cap", "complete_window"]


@dataclass(frozen=True)
class Phase2DelayLook:
    """Compact result for one scheduled analysis."""

    time: float
    enrolled: int
    completed: int
    observed_events: int
    pending: int
    posterior_probability: float
    posterior_probability_mc_se: float
    decision: str
    status: str
    seed: int


@dataclass(frozen=True)
class Phase2DelayCalendarResult:
    """One trial replay, with an immutable tuple of looks."""

    looks: tuple[Phase2DelayLook, ...]
    stopped: bool
    stop_time: float | None
    enrolled: int
    completed_at_stop: int
    completed_at_latest_followup: int
    planned_terminal_time: float
    latest_enrolled_completion_time: float
    final_decision: str


@dataclass(frozen=True)
class Phase2DelayOperatingCharacteristics:
    """Serial Monte Carlo summary; no trial-by-look posterior arrays are kept."""

    trials: int
    stop_rate: float
    stop_rate_mc_se: float
    mean_enrolled: float
    mean_enrolled_mc_se: float
    mean_completed_at_stop: float
    mean_completed_at_stop_mc_se: float
    mean_trial_duration: float
    mean_trial_duration_mc_se: float
    futility_stops: int
    safety_stops: int
    continue_trials: int
    trial_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Phase2DelayCalendarSimulation:
    """Generated replay tape, its seed, and the resulting calendar analysis."""

    seed: int
    arrival_times: NDArray[np.float64]
    latent_event_times: NDArray[np.float64]
    result: Phase2DelayCalendarResult


def _times(
    value: ArrayLike, name: str, *, allow_empty: bool = False, strict: bool = True
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (not allow_empty and not result.size):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    difference = np.diff(result)
    invalid_order = np.any(difference <= 0) if strict else np.any(difference < 0)
    if not np.all(np.isfinite(result)) or np.any(result < 0) or invalid_order:
        order = "strictly increasing" if strict else "nondecreasing"
        raise ValueError(f"{name} must be finite, nonnegative, and {order}")
    return result


def _protocol(
    arrivals: np.ndarray,
    analysis_times: ArrayLike,
    final_analysis: str,
    window: float,
) -> tuple[np.ndarray, float]:
    if final_analysis not in {"at_enrollment_cap", "complete_window"}:
        raise ValueError("final_analysis must be at_enrollment_cap or complete_window")
    if not arrivals.size:
        raise ValueError("at least one planned arrival is required")
    looks = _times(analysis_times, "analysis_times", allow_empty=True)
    terminal = float(arrivals[-1] + (window if final_analysis == "complete_window" else 0.0))
    if not np.isfinite(terminal):
        raise ValueError("planned terminal time is not finite")
    completion = float(arrivals[-1] + window)
    if not np.isfinite(completion) or completion <= arrivals[-1]:
        raise ValueError("latest enrolled completion time is not representable")
    if final_analysis == "complete_window" and terminal != completion:
        raise ValueError("complete-window terminal time is not representable")
    looks = looks[looks <= terminal]
    if not looks.size or looks[-1] != terminal:
        looks = np.append(looks, terminal)
    return looks, terminal


def _look_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _seed_sequence(random_state: int | None) -> np.random.SeedSequence:
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer or None")
    return np.random.SeedSequence(None if random_state is None else int(random_state))


def _preflight(
    n: int, nlooks: int, intervals: int, burn_in: int, draws: int, imputations: int, max_work: int
) -> None:
    # Bound aggregate arrays and worst-case cumulative monitor work before RNG use.
    if nlooks > 10_000:
        raise ValueError("at most 10000 scheduled looks are supported")
    storage = nlooks * (n + intervals + 16) + n
    work = nlooks * (
        (burn_in + draws) * intervals + draws * n * intervals + draws * imputations * n
    )
    if storage > 10_000_000 or work > max_work:
        raise ValueError("requested calendar simulation exceeds max_work")


def _validate_model(
    *,
    endpoint: str,
    threshold: float,
    cutoff: float,
    prior_alpha: float,
    prior_beta: float,
    intervals: int,
    hazard_c: float | ArrayLike,
    lambda0: float,
    burn_in: int,
    hazard_draws: int,
    imputations_per_draw: int,
) -> None:
    if endpoint not in {"response", "toxicity", "progression"}:
        raise ValueError("endpoint must be response, toxicity, or progression")
    threshold = scalar(threshold, "threshold")
    cutoff = scalar(cutoff, "cutoff")
    prior_alpha = scalar(prior_alpha, "prior_alpha")
    prior_beta = scalar(prior_beta, "prior_beta")
    lambda0 = scalar(lambda0, "lambda0")
    if not 0 < threshold < 1 or not 0 < cutoff <= 1:
        raise ValueError("threshold must lie in (0,1) and cutoff in (0,1]")
    if prior_alpha <= 0 or prior_beta <= 0 or not np.isfinite(prior_alpha + prior_beta):
        raise ValueError("prior_alpha and prior_beta must be positive and finite")
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive and finite")
    c = np.broadcast_to(np.asarray(hazard_c, dtype=float), (intervals,))
    if np.any(~np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("hazard_c must be positive and finite")


def _validate_model_for_n(
    n: int, *, intervals: int, burn_in: int, hazard_draws: int, imputations_per_draw: int
) -> None:
    if (
        (burn_in + hazard_draws) * intervals > 200_000
        or hazard_draws * intervals > 2_000_000
        or hazard_draws * n * intervals > 5_000_000
        or hazard_draws * imputations_per_draw * n > 2_000_000
    ):
        raise ValueError("requested Gibbs/imputation run exceeds monitor resource limits")


def _weibull_parameters(
    event_probability: float, late_fraction: float, window: float
) -> tuple[float, float]:
    log_tail_t = log(-np.log1p(-event_probability))
    log_tail_half = log(-np.log1p(-(1.0 - late_fraction) * event_probability))
    shape = (log_tail_t - log_tail_half) / log(2.0)
    log_scale = log(window) - log_tail_t / shape
    with np.errstate(over="ignore", under="ignore"):
        scale = float(np.exp(log_scale))
    if not np.isfinite(shape) or shape <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Weibull calibration is not finite and representable")
    return float(shape), scale


def _weibull_delays(
    uniforms: np.ndarray, event_probability: float, shape: float, scale: float, window: float
) -> np.ndarray:
    delays = np.full(uniforms.shape, np.inf)
    in_window = uniforms <= event_probability
    with np.errstate(over="raise", invalid="raise"):
        delays[in_window] = scale * (-np.log1p(-uniforms[in_window])) ** (1.0 / shape)
    delays[in_window] = np.minimum(delays[in_window], window)
    delays[uniforms == event_probability] = window
    return delays


def replay_phase2_delay_calendar(
    arrival_times: ArrayLike,
    latent_event_times: ArrayLike,
    *,
    analysis_times: ArrayLike,
    final_analysis: FinalAnalysis,
    window: float,
    minimum_completed: int = 5,
    endpoint: str,
    threshold: float,
    cutoff: float,
    prior_alpha: float,
    prior_beta: float,
    intervals: int,
    hazard_c: float | ArrayLike,
    lambda0: float,
    burn_in: int,
    hazard_draws: int,
    imputations_per_draw: int = 1,
    random_state: int | None = None,
    max_work: int = 100_000_000,
) -> Phase2DelayCalendarResult:
    """Replay known enrollment/event times at explicitly scheduled looks.

    Latent event times are measured from each patient's arrival. Positive infinity
    denotes no event in the assessment window. Events and arrivals at a look time
    are processed before that look. Five full windows is the default decision gate.
    """
    window = scalar(window, "window")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be positive and finite")
    arrivals = _times(arrival_times, "arrival_times", strict=False)
    events = np.asarray(latent_event_times, dtype=float)
    if events.ndim != 1 or events.shape != arrivals.shape:
        raise ValueError("latent_event_times must match arrival_times")
    if np.any(np.isnan(events)) or np.any(events < 0):
        raise ValueError("latent_event_times must be nonnegative or infinity")
    finite_in_window = np.isfinite(events) & (events <= window)
    event_deadlines = np.full(events.shape, np.inf)
    event_deadlines[finite_in_window] = arrivals[finite_in_window] + events[finite_in_window]
    if np.any(~np.isfinite(event_deadlines[finite_in_window])) or np.any(
        (events[finite_in_window] > 0)
        & (event_deadlines[finite_in_window] <= arrivals[finite_in_window])
    ):
        raise ValueError("event calendar time is not representable")
    if arrivals.size > 1000:
        raise ValueError("at most 1000 subjects are supported")
    minimum_completed = _integer(minimum_completed, "minimum_completed", 1, 1000)
    intervals = _integer(intervals, "intervals", 1, 100)
    burn_in = _integer(burn_in, "burn_in", 0, 100_000)
    hazard_draws = _integer(hazard_draws, "hazard_draws", 1, 100_000)
    imputations_per_draw = _integer(imputations_per_draw, "imputations_per_draw", 1, 1000)
    max_work = _integer(max_work, "max_work", 1, 2_000_000_000)
    looks, terminal = _protocol(arrivals, analysis_times, final_analysis, window)
    _preflight(
        arrivals.size, len(looks), intervals, burn_in, hazard_draws, imputations_per_draw, max_work
    )
    _validate_model(
        endpoint=endpoint,
        threshold=threshold,
        cutoff=cutoff,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        intervals=intervals,
        hazard_c=hazard_c,
        lambda0=lambda0,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    _validate_model_for_n(
        arrivals.size,
        intervals=intervals,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    seed_sequences = _seed_sequence(random_state).spawn(len(looks))
    records: list[Phase2DelayLook] = []
    stopped = False
    stop_time: float | None = None
    stop_enrolled = 0
    stop_completed = 0
    for now, seed_sequence in zip(looks, seed_sequences):
        enrolled = int(np.searchsorted(arrivals, now, side="right"))
        deadlines = arrivals[:enrolled] + window
        completed_mask = deadlines <= now
        # Compare absolute completion timestamps to avoid subtraction rounding
        # changing the full-window gate at non-binary calendar times.
        ages = np.maximum(now - arrivals[:enrolled], 0.0)
        ages = np.where(completed_mask, window, np.minimum(ages, np.nextafter(window, 0.0)))
        observed_event = event_deadlines[:enrolled] <= now
        observed_time = np.where(observed_event, events[:enrolled], 0.0)
        followup = np.where(observed_event, events[:enrolled], np.minimum(ages, window))
        completed = int(np.count_nonzero(completed_mask))
        seed = _look_seed(seed_sequence)
        probability = float("nan")
        mc_se = float("nan")
        decision = "continue"
        status = "gate_not_reached"
        if completed >= minimum_completed:
            status = "analyzed"
            monitored = phase2_delay_monitor(
                observed_event.astype(int),
                observed_time,
                followup,
                window=window,
                endpoint=endpoint,
                threshold=threshold,
                cutoff=cutoff,
                intervals=intervals,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                hazard_c=hazard_c,
                lambda0=lambda0,
                burn_in=burn_in,
                hazard_draws=hazard_draws,
                imputations_per_draw=imputations_per_draw,
                seed=seed,
            )
            probability = monitored.posterior_probability
            mc_se = monitored.posterior_probability_mc_se
            decision = monitored.decision
            del monitored
        records.append(
            Phase2DelayLook(
                float(now),
                enrolled,
                completed,
                int(np.count_nonzero(observed_event)),
                enrolled
                - int(np.count_nonzero(observed_event))
                - int(np.count_nonzero((~observed_event) & (ages >= window))),
                probability,
                mc_se,
                decision,
                status,
                seed,
            )
        )
        if decision != "continue":
            stopped = True
            stop_time = float(now)
            stop_enrolled = enrolled
            stop_completed = completed
            break
    final = records[-1]
    followup_enrolled = (
        stop_enrolled if stopped else int(np.searchsorted(arrivals, terminal, side="right"))
    )
    latest_completion = (
        float(np.max(arrivals[:followup_enrolled]) + window) if followup_enrolled else terminal
    )
    if not np.isfinite(latest_completion):
        raise ValueError("latest enrolled completion time is not finite")
    followup_completed = int(
        np.count_nonzero(arrivals[:followup_enrolled] + window <= latest_completion)
    )
    return Phase2DelayCalendarResult(
        tuple(records),
        stopped,
        stop_time,
        stop_enrolled if stopped else final.enrolled,
        stop_completed if stopped else final.completed,
        followup_completed,
        terminal,
        latest_completion,
        final.decision,
    )


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    return mean, se


def simulate_phase2_delay_calendar(
    *,
    event_probability: float,
    late_fraction: float,
    accrual_rate: float = 2.0,
    max_subjects: int = 50,
    analysis_times: ArrayLike,
    final_analysis: FinalAnalysis,
    window: float,
    minimum_completed: int = 5,
    endpoint: str,
    threshold: float,
    cutoff: float,
    prior_alpha: float,
    prior_beta: float,
    intervals: int,
    hazard_c: float | ArrayLike,
    lambda0: float,
    burn_in: int,
    hazard_draws: int,
    imputations_per_draw: int = 1,
    random_state: int | None = None,
    max_work: int = 100_000_000,
) -> Phase2DelayCalendarSimulation:
    """Generate one exponential-accrual/Weibull-event tape and replay it.

    The Weibull calibration sets F(window)=event_probability and the fraction of
    events in (window/2, window] to late_fraction. First enrollment is at time 0.
    """
    p = scalar(event_probability, "event_probability")
    q = scalar(late_fraction, "late_fraction")
    rate = scalar(accrual_rate, "accrual_rate")
    window = scalar(window, "window")
    if not 0 < p < 1 or not 0 < q < 1 or rate <= 0 or window <= 0:
        raise ValueError("event_probability, late_fraction, accrual_rate and window must be valid")
    n = _integer(max_subjects, "max_subjects", 1, 1000)
    # Solve F(T/2)=(1-q)F(T), then derive shape and scale.
    shape, scale = _weibull_parameters(p, q, window)
    looks = _times(analysis_times, "analysis_times", allow_empty=True)
    if final_analysis not in {"at_enrollment_cap", "complete_window"}:
        raise ValueError("final_analysis must be at_enrollment_cap or complete_window")
    intervals = _integer(intervals, "intervals", 1, 100)
    burn_in = _integer(burn_in, "burn_in", 0, 100_000)
    hazard_draws = _integer(hazard_draws, "hazard_draws", 1, 100_000)
    imputations_per_draw = _integer(imputations_per_draw, "imputations_per_draw", 1, 1000)
    max_work = _integer(max_work, "max_work", 1, 2_000_000_000)
    _preflight(n, len(looks) + 1, intervals, burn_in, hazard_draws, imputations_per_draw, max_work)
    minimum_completed = _integer(minimum_completed, "minimum_completed", 1, 1000)
    _validate_model(
        endpoint=endpoint,
        threshold=threshold,
        cutoff=cutoff,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        intervals=intervals,
        hazard_c=hazard_c,
        lambda0=lambda0,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    _validate_model_for_n(
        n,
        intervals=intervals,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    phase_seed = _seed_sequence(random_state)
    trial_seed = int(random_state) if random_state is not None else _look_seed(phase_seed)
    trial_rng = np.random.default_rng(trial_seed)
    arrivals = np.empty(n, dtype=float)
    arrivals[0] = 0.0
    if n > 1:
        arrivals[1:] = np.cumsum(trial_rng.exponential(1.0 / rate, n - 1))
    if not np.all(np.isfinite(arrivals)):
        raise ValueError("generated accrual times are not finite")
    u = trial_rng.random(n)
    latent = _weibull_delays(u, p, shape, scale, window)
    replay = replay_phase2_delay_calendar(
        arrivals,
        latent,
        analysis_times=analysis_times,
        final_analysis=final_analysis,
        window=window,
        minimum_completed=minimum_completed,
        endpoint=endpoint,
        threshold=threshold,
        cutoff=cutoff,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        intervals=intervals,
        hazard_c=hazard_c,
        lambda0=lambda0,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
        random_state=trial_seed,
        max_work=max_work,
    )
    arrivals.flags.writeable = False
    latent.flags.writeable = False
    return Phase2DelayCalendarSimulation(trial_seed, arrivals, latent, replay)


def simulate_phase2_delay_calendar_oc(
    *,
    trials: int,
    event_probability: float,
    late_fraction: float,
    accrual_rate: float = 2.0,
    max_subjects: int = 50,
    analysis_times: ArrayLike,
    final_analysis: FinalAnalysis,
    window: float,
    minimum_completed: int = 5,
    endpoint: str,
    threshold: float,
    cutoff: float,
    prior_alpha: float,
    prior_beta: float,
    intervals: int,
    hazard_c: float | ArrayLike,
    lambda0: float,
    burn_in: int,
    hazard_draws: int,
    imputations_per_draw: int = 1,
    random_state: int | None = None,
    max_work: int = 100_000_000,
) -> Phase2DelayOperatingCharacteristics:
    """Run a serial operating-characteristic batch with bounded summary storage."""
    trials = _integer(trials, "trials", 1, 100_000)
    max_work = _integer(max_work, "max_work", 1, 2_000_000_000)
    if max_work < trials:
        raise ValueError("max_work must be at least trials")
    if trials * 4 > 10_000_000:
        raise ValueError("requested OC summary storage is too large")
    n = _integer(max_subjects, "max_subjects", 1, 1000)
    p = scalar(event_probability, "event_probability")
    q = scalar(late_fraction, "late_fraction")
    rate = scalar(accrual_rate, "accrual_rate")
    window = scalar(window, "window")
    if not 0 < p < 1 or not 0 < q < 1 or rate <= 0 or window <= 0:
        raise ValueError("event_probability, late_fraction, accrual_rate and window must be valid")
    _weibull_parameters(p, q, window)
    analysis = _times(analysis_times, "analysis_times", allow_empty=True)
    if final_analysis not in {"at_enrollment_cap", "complete_window"}:
        raise ValueError("final_analysis must be at_enrollment_cap or complete_window")
    intervals = _integer(intervals, "intervals", 1, 100)
    burn_in = _integer(burn_in, "burn_in", 0, 100_000)
    hazard_draws = _integer(hazard_draws, "hazard_draws", 1, 100_000)
    imputations_per_draw = _integer(imputations_per_draw, "imputations_per_draw", 1, 1000)
    _preflight(
        n,
        len(analysis) + 1,
        intervals,
        burn_in,
        hazard_draws,
        imputations_per_draw,
        max_work // trials,
    )
    minimum_completed = _integer(minimum_completed, "minimum_completed", 1, 1000)
    _validate_model(
        endpoint=endpoint,
        threshold=threshold,
        cutoff=cutoff,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        intervals=intervals,
        hazard_c=hazard_c,
        lambda0=lambda0,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    _validate_model_for_n(
        n,
        intervals=intervals,
        burn_in=burn_in,
        hazard_draws=hazard_draws,
        imputations_per_draw=imputations_per_draw,
    )
    _seed_sequence(random_state)
    base_seed = int(random_state) if random_state is not None else _look_seed(_seed_sequence(None))
    seeds = tuple(_look_seed(np.random.SeedSequence([base_seed, i])) for i in range(trials))
    stop_flags = np.empty(trials)
    enrolled = np.empty(trials)
    completed = np.empty(trials)
    duration = np.empty(trials)
    stops = {"stop_futility": 0, "stop_safety": 0, "continue": 0}
    for i, seed in enumerate(seeds):
        trial = simulate_phase2_delay_calendar(
            event_probability=event_probability,
            late_fraction=late_fraction,
            accrual_rate=accrual_rate,
            max_subjects=max_subjects,
            analysis_times=analysis_times,
            final_analysis=final_analysis,
            window=window,
            minimum_completed=minimum_completed,
            endpoint=endpoint,
            threshold=threshold,
            cutoff=cutoff,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            intervals=intervals,
            hazard_c=hazard_c,
            lambda0=lambda0,
            burn_in=burn_in,
            hazard_draws=hazard_draws,
            imputations_per_draw=imputations_per_draw,
            random_state=seed,
            max_work=max_work // trials,
        )
        result = trial.result
        stop_flags[i] = result.stopped
        enrolled[i] = result.enrolled
        completed[i] = result.completed_at_stop
        duration[i] = result.stop_time if result.stopped else result.planned_terminal_time
        stops[result.final_decision] += 1
    stop_rate, stop_se = _mean_se(stop_flags)
    mean_n, mean_n_se = _mean_se(enrolled)
    mean_completed, mean_completed_se = _mean_se(completed)
    mean_duration, mean_duration_se = _mean_se(duration)
    return Phase2DelayOperatingCharacteristics(
        trials,
        stop_rate,
        stop_se,
        mean_n,
        mean_n_se,
        mean_completed,
        mean_completed_se,
        mean_duration,
        mean_duration_se,
        stops["stop_futility"],
        stops["stop_safety"],
        stops["continue"],
        seeds,
    )
