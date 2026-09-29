"""Source-informed serial simulation and operating characteristics for ARAND."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .arand_calendar import (
    ArandCalendarTrial,
    ArandControllerPolicy,
    _validate_policy,
    arand_calendar_replay,
)

_MAX_TRIALS = 10_000
_MAX_CANDIDATES = 10_000
_MAX_CANDIDATE_ARM_CELLS = 2_000_000
_MAX_CALENDAR_STORAGE_CELLS = 10_000_000
_MAX_AGGREGATE_WORK = 2_000_000_000
_MAX_AGGREGATE_STORAGE = 2_000_000_000


def _readonly(values: ArrayLike, dtype: object | None = None) -> NDArray:
    array = np.ascontiguousarray(cast(Any, values), dtype=cast(Any, dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _nonnegative_seed(seed: int, name: str = "seed") -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(seed)


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    number = scalar(value, name)
    if number != np.floor(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return int(number)


@dataclass(frozen=True)
class ArandSimulationConfig:
    """Explicit ARAND trial settings, excluding generated data and accrual.

    ``prior`` contains one row per arm. Binary rows are beta shapes; exponential
    rows are inverse-gamma shape/scale pairs. ``parameter`` selects whether an
    exponential scenario parameter is a true mean or median, in the same time
    units as accrual, observation, and follow-up settings. ``policy`` is
    required because the guide does not settle all controller choices.
    """

    prior: ArrayLike
    family: Literal["binary", "exponential"]
    analysis_times: ArrayLike
    max_enrollment: int | None
    max_duration: float | None
    final_followup: float
    minimum_enrollment: int
    policy: ArandControllerPolicy
    binary_window: float | None = None
    parameter: Literal["mean", "median"] = "mean"
    maximize: bool = True
    initial_equal_randomization: int = 0
    tuning: float = 0.5
    minimum_allocation: float = 0.0
    loser_cutoff: float | None = None
    early_winner_cutoff: float | None = None
    final_winner_cutoff: float | None = None
    futility_threshold: float | None = None
    futility_cutoff: float | None = None
    absolute_tolerance: float = 1e-9
    max_work: int = 100_000_000


@dataclass(frozen=True)
class ArandSimulationResult:
    """Compact serial OC summary; individual histories are discarded.

    ``selected_*`` uses all completed simulation calls as its denominator,
    including trials with no winner or no enrollment. The `selected` event is
    the arm returned as the early winner when the trial stopped early,
    otherwise the final winner. The distinct suspension, futility, and early-
    winner-displacement events are not collapsed into a native ``dropped``
    measure because the guide does not define that aggregation.

    Patient-count intervals are empirical 2.5th and 97.5th percentiles using
    NumPy's linear (Hyndman-Fan type 7) method. ``trial_seeds[i]`` reproduces
    replicate i with :func:`simulate_arand_trial` and the same configuration.
    """

    trials: int
    family: str
    scenario_parameters: FloatArray
    trial_seeds: NDArray[np.uint64]
    selected_count: NDArray[np.int64]
    selected_probability: FloatArray
    selected_mcse: FloatArray
    early_selected_count: NDArray[np.int64]
    early_selected_probability: FloatArray
    early_selected_mcse: FloatArray
    final_selected_count: NDArray[np.int64]
    final_selected_probability: FloatArray
    final_selected_mcse: FloatArray
    ever_suspended_count: NDArray[np.int64]
    ever_suspended_probability: FloatArray
    ever_suspended_mcse: FloatArray
    permanently_futile_count: NDArray[np.int64]
    permanently_futile_probability: FloatArray
    permanently_futile_mcse: FloatArray
    early_winner_displaced_count: NDArray[np.int64]
    early_winner_displaced_probability: FloatArray
    early_winner_displaced_mcse: FloatArray
    mean_patients_per_arm: FloatArray
    patients_per_arm_mcse: FloatArray
    patients_per_arm_interval: FloatArray
    no_winner_count: int
    no_winner_probability: float
    no_winner_mcse: float
    stop_reasons: tuple[str, ...]
    stop_reason_count: NDArray[np.int64]
    stop_reason_probability: FloatArray
    stop_reason_mcse: FloatArray
    mean_decision_duration: float
    decision_duration_mcse: float
    mean_completed_duration: float
    completed_duration_mcse: float
    maximum_candidate_arrivals: int
    estimated_total_work: int
    estimated_peak_storage_bytes: int


def _prepared(
    config: ArandSimulationConfig,
    scenario_parameters: ArrayLike,
    accrual_rate: float,
    max_candidate_arrivals: int,
) -> tuple[FloatArray, FloatArray, float, int, int, int, int]:
    if not isinstance(config, ArandSimulationConfig):
        raise TypeError("config must be an ArandSimulationConfig")
    _validate_policy(config.policy)
    if config.family not in {"binary", "exponential"}:
        raise ValueError("family must be binary or exponential")
    if config.parameter not in {"mean", "median"}:
        raise ValueError("parameter must be mean or median")
    if not isinstance(config.maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")

    prior_raw = np.asarray(config.prior)
    if prior_raw.ndim != 2 or not 1 <= prior_raw.shape[0] <= 10 or prior_raw.shape[1] != 2:
        raise ValueError("prior must contain 1 to 10 rows of two positive parameters")
    prior = finite(prior_raw, "prior")
    if np.any(prior <= 0):
        raise ValueError("prior must contain 1 to 10 rows of two positive parameters")
    if config.family == "binary" and not np.all(np.isfinite(prior.sum(axis=1))):
        raise ValueError("beta prior shape sums must be finite")
    truth_raw = np.asarray(scenario_parameters)
    if truth_raw.ndim != 1 or truth_raw.shape != (len(prior),):
        raise ValueError("scenario_parameters must contain one value per arm")
    truth = finite(truth_raw, "scenario_parameters")
    if truth.ndim != 1 or truth.shape != (len(prior),):
        raise ValueError("scenario_parameters must contain one value per arm")
    if config.family == "binary":
        if np.any((truth < 0) | (truth > 1)):
            raise ValueError("binary scenario probabilities must lie in [0,1]")
        if config.binary_window is None:
            raise ValueError("binary scenarios require binary_window")
        window = scalar(config.binary_window, "binary_window")
        if window <= 0:
            raise ValueError("binary_window must be positive")
    else:
        if np.any(truth <= 0):
            raise ValueError("exponential scenario mean/median parameters must be positive")
        if config.binary_window is not None:
            raise ValueError("exponential scenarios do not use binary_window")
        window = 0.0

    candidate_count = _integer(max_candidate_arrivals, "max_candidate_arrivals", 1, _MAX_CANDIDATES)
    arms = len(prior)
    candidate_cells = candidate_count * arms
    if candidate_cells > _MAX_CANDIDATE_ARM_CELLS:
        raise ValueError("candidate-by-arm potential outcomes exceed two million cells")

    looks_raw = np.asarray(config.analysis_times)
    if looks_raw.ndim != 1 or looks_raw.size > 10_000:
        raise ValueError("analysis_times must be a strictly increasing nonnegative vector")
    arrivals = finite(looks_raw, "analysis_times")
    if np.any(arrivals < 0) or np.any(np.diff(arrivals) <= 0):
        raise ValueError("analysis_times must be a strictly increasing nonnegative vector")
    if config.max_enrollment is None and config.max_duration is None:
        raise ValueError("max_enrollment or max_duration must be specified")
    if config.max_enrollment is not None:
        _integer(config.max_enrollment, "max_enrollment", 1, candidate_count)
    if config.max_duration is not None and scalar(config.max_duration, "max_duration") <= 0:
        raise ValueError("max_duration must be positive")
    if scalar(config.final_followup, "final_followup") < 0:
        raise ValueError("final_followup must be nonnegative")
    _integer(config.minimum_enrollment, "minimum_enrollment", 1, 100_000)
    _integer(config.initial_equal_randomization, "initial_equal_randomization", 0, 100_000)
    if config.initial_equal_randomization > candidate_count:
        raise ValueError("initial_equal_randomization cannot exceed max_candidate_arrivals")
    if config.max_enrollment is not None and config.minimum_enrollment > config.max_enrollment:
        raise ValueError("minimum_enrollment cannot exceed max_enrollment")
    if (
        config.max_enrollment is not None
        and config.initial_equal_randomization > config.max_enrollment
    ):
        raise ValueError("initial_equal_randomization cannot exceed max_enrollment")
    tuning = scalar(config.tuning, "tuning")
    floor = scalar(config.minimum_allocation, "minimum_allocation")
    if tuning < 0 or not 0 <= floor < 1:
        raise ValueError("tuning must be nonnegative and minimum_allocation in [0,1)")
    tolerance = scalar(config.absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("absolute_tolerance must lie in [1e-12,1e-3]")
    _integer(config.max_work, "max_work", 1, 2_000_000_000)
    for name, cutoff in (
        ("loser_cutoff", config.loser_cutoff),
        ("early_winner_cutoff", config.early_winner_cutoff),
        ("final_winner_cutoff", config.final_winner_cutoff),
        ("futility_cutoff", config.futility_cutoff),
    ):
        if cutoff is not None and not 0 <= scalar(cutoff, name) <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    if (config.futility_threshold is None) != (config.futility_cutoff is None):
        raise ValueError("futility_threshold and futility_cutoff must be supplied together")
    if config.futility_threshold is not None:
        threshold = scalar(config.futility_threshold, "futility_threshold")
        if (config.family == "binary" and not 0 <= threshold <= 1) or (
            config.family == "exponential" and threshold < 0
        ):
            raise ValueError("futility_threshold is outside the outcome parameter domain")
        if not config.maximize:
            raise ValueError("futility is available only when maximizing")

    rate = scalar(accrual_rate, "accrual_rate")
    if rate <= 0:
        raise ValueError("accrual_rate must be positive")
    mean_gap = 1.0 / rate
    if not np.isfinite(mean_gap) or mean_gap <= 0:
        raise ValueError("accrual_rate must give a finite, positive mean arrival gap")
    # Match the calendar replay's candidate, look-storage and per-trial work
    # limits, using the configured cap as the worst case before any draws.
    look_bound = candidate_count + len(arrivals) + 1
    work = look_bound * candidate_count * arms + look_bound * arms * max(1, arms - 1) * 500
    storage_cells = look_bound * (arms * 48 + 1_024)
    if look_bound > 10_000 or storage_cells > _MAX_CALENDAR_STORAGE_CELLS:
        raise ValueError("candidate cap exceeds calendar look/storage limits")
    if work > int(config.max_work):
        raise ValueError("config.max_work is below the worst-case single-trial estimate")
    return (
        _readonly(prior, np.float64),
        _readonly(truth, np.float64),
        rate,
        candidate_count,
        arms,
        int(work),
        int(storage_cells),
    )


def _candidate_arrivals(
    rng: np.random.Generator,
    *,
    rate: float,
    maximum: int,
) -> FloatArray:
    gaps = rng.exponential(scale=1.0 / rate, size=maximum)
    if np.any(~np.isfinite(gaps)):
        raise ArithmeticError("Poisson inter-arrival times are not representable")
    arrivals = np.cumsum(gaps, dtype=np.float64)
    if np.any(~np.isfinite(arrivals)):
        raise ArithmeticError("Poisson arrival times are not representable")
    # Keep the bounded full tape. The calendar replay stops at its duration
    # horizon, while retaining all candidate rows avoids changing its enrollment
    # cap contract when few Poisson arrivals occur before that horizon.
    return _readonly(arrivals)


def _trial_seed_streams(seed: int) -> tuple[np.random.Generator, ...]:
    return tuple(np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3))


def _calendar_arguments(config: ArandSimulationConfig) -> dict[str, object]:
    return {
        "prior": config.prior,
        "family": config.family,
        "analysis_times": config.analysis_times,
        "max_enrollment": config.max_enrollment,
        "max_duration": config.max_duration,
        "final_followup": config.final_followup,
        "minimum_enrollment": config.minimum_enrollment,
        "parameter": config.parameter,
        "maximize": config.maximize,
        "initial_equal_randomization": config.initial_equal_randomization,
        "tuning": config.tuning,
        "minimum_allocation": config.minimum_allocation,
        "loser_cutoff": config.loser_cutoff,
        "early_winner_cutoff": config.early_winner_cutoff,
        "final_winner_cutoff": config.final_winner_cutoff,
        "futility_threshold": config.futility_threshold,
        "futility_cutoff": config.futility_cutoff,
        "policy": config.policy,
        "absolute_tolerance": config.absolute_tolerance,
        "max_work": config.max_work,
    }


def simulate_arand_trial(
    config: ArandSimulationConfig,
    scenario_parameters: ArrayLike,
    *,
    accrual_rate: float,
    max_candidate_arrivals: int,
    seed: int,
) -> ArandCalendarTrial:
    """Generate and replay one Poisson-accrual ARAND trial.

    The first arrival is exponential from time zero, as are all subsequent
    inter-arrival gaps. Binary scenario values are arm response/event
    probabilities and use the configured fixed observation window. Exponential
    scenario values are true means or medians, selected by ``config.parameter``.
    Separate child random streams generate arrivals, potential outcomes and
    assignments so the returned ``data_seed`` reproduces this trial directly.
    """
    seed_value = _nonnegative_seed(seed)
    prior, truth, rate, candidates, arms, _, _ = _prepared(
        config, scenario_parameters, accrual_rate, max_candidate_arrivals
    )
    del prior
    arrival_rng, outcome_rng, assignment_rng = _trial_seed_streams(seed_value)
    arrivals = _candidate_arrivals(
        arrival_rng,
        rate=rate,
        maximum=candidates,
    )
    shape = (len(arrivals), arms)
    if config.family == "binary":
        outcomes = outcome_rng.binomial(1, truth[None, :], size=shape).astype(np.int8)
        replay_inputs: dict[str, object] = {
            "binary_outcomes": outcomes,
            "binary_window": config.binary_window,
        }
    else:
        scale = truth if config.parameter == "mean" else truth / np.log(2.0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(
                "scenario mean/median cannot be converted to a finite exponential scale"
            )
        event_times = outcome_rng.exponential(scale=scale[None, :], size=shape)
        if np.any(~np.isfinite(event_times)):
            raise ArithmeticError("generated exponential event times are not representable")
        replay_inputs = {"event_times": event_times}
    uniforms = assignment_rng.random(len(arrivals))
    result = arand_calendar_replay(
        arrivals,
        uniforms,
        **cast(Any, _calendar_arguments(config)),
        **cast(Any, replay_inputs),
        data_seed=seed_value,
    )
    reached_enrollment_limit = (
        config.max_enrollment is not None and len(result.assignments) >= config.max_enrollment
    )
    early_stop_covered = (
        result.stopped_early and result.stop_time is not None and result.stop_time <= arrivals[-1]
    )
    duration_covered = (
        config.max_duration is not None
        and config.max_duration <= arrivals[-1]
        and (
            config.policy.duration_minimum_precedence == "duration_wins"
            or len(result.assignments) >= config.minimum_enrollment
        )
    )
    if not (reached_enrollment_limit or early_stop_covered or duration_covered):
        if result.reason == "minimum_not_reached_input_exhausted":
            raise RuntimeError("candidate-arrival budget exhausted before minimum enrollment")
        raise RuntimeError("candidate-arrival budget exhausted before trial completion")
    return result


def _bernoulli_summary(counts: NDArray[np.int64], trials: int) -> tuple[FloatArray, FloatArray]:
    probability = counts.astype(float) / trials
    if trials == 1:
        mcse = np.full(probability.shape, np.nan)
    else:
        mcse = np.sqrt(probability * (1.0 - probability) / trials)
    return probability, mcse


def _mean_summary(total: NDArray[np.float64], squares: NDArray[np.float64], trials: int):
    mean = total / trials
    if trials < 2:
        return mean, np.full(mean.shape, np.nan)
    variance = np.maximum((squares - total * total / trials) / (trials - 1), 0.0)
    return mean, np.sqrt(variance / trials)


def _scaled_mean_mcse(values: NDArray[np.float64]) -> tuple[float, float]:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, float("nan") if values.size == 1 else 0.0
    normalized = values / scale
    mean_normalized = float(np.mean(normalized))
    mean = scale * mean_normalized
    if values.size == 1:
        mcse = float("nan")
    else:
        standard_deviation_normalized = float(np.std(normalized, ddof=1))
        mcse = scale * standard_deviation_normalized / np.sqrt(values.size)
    if not np.isfinite(mean) or (not np.isnan(mcse) and not np.isfinite(mcse)):
        raise ArithmeticError("duration summary is not representable")
    return mean, mcse


def simulate_arand(
    config: ArandSimulationConfig,
    scenario_parameters: ArrayLike,
    *,
    accrual_rate: float,
    max_candidate_arrivals: int,
    trials: int,
    seed: int,
    max_total_work: int = 500_000_000,
    max_total_storage_bytes: int = 512_000_000,
) -> ArandSimulationResult:
    """Run bounded serial ARAND trials and return per-arm operating characteristics.

    All trial seeds and the worst-case workload/storage are checked before
    deriving replicate seeds. Histories are discarded after each replay; only
    trial seeds, arm-level assignment counts for percentile calculation and
    aggregate summaries are retained.
    """
    seed_value = _nonnegative_seed(seed)
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    total_work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_AGGREGATE_WORK)
    total_storage_limit = _integer(
        max_total_storage_bytes, "max_total_storage_bytes", 1, _MAX_AGGREGATE_STORAGE
    )
    prior, truth, _, candidate_count, arms, per_trial_work, per_trial_storage_cells = _prepared(
        config, scenario_parameters, accrual_rate, max_candidate_arrivals
    )
    worst_work = trial_count * per_trial_work
    # Counts are retained for empirical quantiles; each replicate's event,
    # timing and calendar tapes are transient and processed serially.
    summary_bytes = trial_count * (8 + 8 * arms + 8 * arms + 16)
    single_trial_bytes = (
        candidate_count * arms * 32
        + candidate_count * 128
        + per_trial_storage_cells * 8
        + np.asarray(config.analysis_times).size * 8
        + 16_384
    )
    peak_storage = summary_bytes + single_trial_bytes
    if worst_work > total_work_limit:
        raise ValueError("max_total_work is below the aggregate worst-case estimate")
    if peak_storage > total_storage_limit:
        raise ValueError("max_total_storage_bytes is below the aggregate peak estimate")

    seed_sequence = np.random.SeedSequence(seed_value)
    trial_seeds = np.empty(trial_count, dtype=np.uint64)
    for index in range(trial_count):
        child = seed_sequence.spawn(1)[0]
        trial_seeds[index] = child.generate_state(1, dtype=np.uint64)[0]
    patient_counts = np.zeros((trial_count, arms), dtype=np.int64)
    selection = np.zeros(arms, dtype=np.int64)
    early = np.zeros(arms, dtype=np.int64)
    final = np.zeros(arms, dtype=np.int64)
    suspended = np.zeros(arms, dtype=np.int64)
    futile = np.zeros(arms, dtype=np.int64)
    displaced = np.zeros(arms, dtype=np.int64)
    no_winner = 0
    reason_counts: dict[str, int] = {}
    decision_durations = np.empty(trial_count, dtype=np.float64)
    completed_durations = np.empty(trial_count, dtype=np.float64)

    for index, seed_item in enumerate(trial_seeds):
        result = simulate_arand_trial(
            config,
            truth,
            accrual_rate=accrual_rate,
            max_candidate_arrivals=candidate_count,
            seed=int(seed_item),
        )
        counts = np.bincount(result.assignments, minlength=arms).astype(np.int64)
        patient_counts[index] = counts
        early_winner = result.early_winner
        final_winner = result.final_winner
        selected_winner = early_winner if early_winner is not None else final_winner
        if selected_winner is None:
            no_winner += 1
        else:
            selection[selected_winner] += 1
        if early_winner is not None:
            early[early_winner] += 1
            displaced += np.arange(arms) != early_winner
        if final_winner is not None:
            final[final_winner] += 1
        ever_suspended = np.array(result.suspended, dtype=bool, copy=True)
        ever_futile = np.array(result.futile, dtype=bool, copy=True)
        for look in result.looks:
            ever_suspended |= look.suspended
            ever_futile |= look.futile
        suspended += ever_suspended
        futile += ever_futile
        reason_counts[result.reason] = reason_counts.get(result.reason, 0) + 1
        decision_durations[index] = result.decision_duration
        completed_durations[index] = result.completed_duration
        # Do not keep a full replay history alive while the next one is built.
        del result

    selected_p, selected_e = _bernoulli_summary(selection, trial_count)
    early_p, early_e = _bernoulli_summary(early, trial_count)
    final_p, final_e = _bernoulli_summary(final, trial_count)
    suspended_p, suspended_e = _bernoulli_summary(suspended, trial_count)
    futile_p, futile_e = _bernoulli_summary(futile, trial_count)
    displaced_p, displaced_e = _bernoulli_summary(displaced, trial_count)
    patients_total = patient_counts.sum(axis=0, dtype=np.float64)
    patients_squares = np.square(patient_counts).sum(axis=0, dtype=np.float64)
    mean_patients, patients_mcse = _mean_summary(patients_total, patients_squares, trial_count)
    patients_interval = np.quantile(patient_counts, [0.025, 0.975], axis=0, method="linear").T
    reason_names = tuple(sorted(reason_counts))
    reason_count_array = np.asarray([reason_counts[name] for name in reason_names], dtype=np.int64)
    reason_p, reason_e = _bernoulli_summary(reason_count_array, trial_count)
    no_winner_p, no_winner_e = _bernoulli_summary(
        np.asarray([no_winner], dtype=np.int64), trial_count
    )
    decision_mean, decision_error = _scaled_mean_mcse(decision_durations)
    completed_mean, completed_error = _scaled_mean_mcse(completed_durations)
    return ArandSimulationResult(
        trial_count,
        config.family,
        _readonly(truth, np.float64),
        _readonly(trial_seeds, np.uint64),
        _readonly(selection, np.int64),
        _readonly(selected_p),
        _readonly(selected_e),
        _readonly(early, np.int64),
        _readonly(early_p),
        _readonly(early_e),
        _readonly(final, np.int64),
        _readonly(final_p),
        _readonly(final_e),
        _readonly(suspended, np.int64),
        _readonly(suspended_p),
        _readonly(suspended_e),
        _readonly(futile, np.int64),
        _readonly(futile_p),
        _readonly(futile_e),
        _readonly(displaced, np.int64),
        _readonly(displaced_p),
        _readonly(displaced_e),
        _readonly(mean_patients),
        _readonly(patients_mcse),
        _readonly(patients_interval),
        no_winner,
        float(no_winner_p[0]),
        float(no_winner_e[0]),
        reason_names,
        _readonly(reason_count_array, np.int64),
        _readonly(reason_p),
        _readonly(reason_e),
        decision_mean,
        decision_error,
        completed_mean,
        completed_error,
        candidate_count,
        int(worst_work),
        int(peak_storage),
    )
