"""Bounded serial timing simulation and duration summaries for Multc Lean."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .multc_calendar import MultcCalendarTrial, run_multc_calendar_trial
from .multc_core import MultcLeanDesign

_MAX_TRIALS = 10_000
_MAX_TOTAL_WORK = 2_000_000_000
_MAX_TOTAL_STORAGE = 2_000_000_000


def _owned(values: ArrayLike, dtype: type | None = None) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _seed_value(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _positive_integer(value: int, name: str, maximum: int) -> int:
    number = scalar(value, name)
    if number != np.floor(number) or not 1 <= number <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(number)


def _scaled_mean_mcse(values: NDArray[np.float64]) -> tuple[float, float]:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, float("nan") if values.size == 1 else 0.0
    normalized = values / scale
    mean = scale * float(np.mean(normalized))
    error = (
        float("nan")
        if values.size == 1
        else scale * float(np.std(normalized, ddof=1)) / np.sqrt(values.size)
    )
    if not np.isfinite(mean) or (not np.isnan(error) and not np.isfinite(error)):
        raise ArithmeticError("simulation duration summary is not representable")
    return mean, error


def _bounded_mean_mcse(values: NDArray[np.float64]) -> tuple[float, float]:
    mean = float(np.mean(values))
    error = (
        float("nan")
        if values.size == 1
        else float(np.std(values, ddof=1) / np.sqrt(values.size))
    )
    return mean, error


@dataclass(frozen=True)
class MultcSimulationConfig:
    """Explicit timing assumptions for one supported Multc Lean design.

    Positive response times follow an exponential law calibrated to have 95%
    probability by ``response_window``. ``response_timing`` makes the unresolved
    meaning of native truncation explicit: conditional truncation or clipping at
    the window. Nonresponse becomes known at the window. Toxicity availability
    uses the caller-supplied fixed delay because the guide does not specify its
    time law. The calendar replay's accrual-open-clock convention is retained.
    """

    design: MultcLeanDesign
    response_window: float
    toxicity_delay: float
    response_timing: Literal["conditional_truncated_exponential", "clip_at_window"]


@dataclass(frozen=True)
class MultcSimulationResult:
    """Aggregate operating characteristics; no patient histories are retained.

    Means and decision probabilities use every completed replicate, including
    trials stopped before enrollment. Duration intervals use NumPy's linear
    (Hyndman-Fan type 7) empirical 2.5th and 97.5th percentiles. MCSEs are
    undefined for a single replicate.
    """

    trials: int
    joint_probabilities: FloatArray
    trial_seeds: NDArray[np.uint64]
    mean_enrolled: float
    enrolled_mcse: float
    enrollment_probability: FloatArray
    mean_responses: float
    responses_mcse: float
    mean_toxicities: float
    toxicities_mcse: float
    mean_duration: float
    duration_mcse: float
    duration_interval: FloatArray
    mean_accrual_duration: float
    accrual_duration_mcse: float
    accrual_duration_interval: FloatArray
    mean_paused_duration: float
    paused_duration_mcse: float
    mean_last_followup_time: float
    last_followup_time_mcse: float
    decisions: tuple[str, ...]
    decision_count: NDArray[np.int64]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    maximum_work_per_trial: int
    estimated_total_work: int
    estimated_peak_storage_bytes: int


def _validate_config(config: MultcSimulationConfig) -> tuple[MultcLeanDesign, float, float]:
    if not isinstance(config, MultcSimulationConfig):
        raise TypeError("config must be a MultcSimulationConfig")
    if not isinstance(config.design, MultcLeanDesign):
        raise TypeError("config.design must be a MultcLeanDesign")
    if config.response_timing not in {
        "conditional_truncated_exponential",
        "clip_at_window",
    }:
        raise ValueError("response_timing must select an explicit truncation convention")
    window = scalar(config.response_window, "response_window")
    toxicity_delay = scalar(config.toxicity_delay, "toxicity_delay")
    if window <= 0:
        raise ValueError("response_window must be positive")
    if toxicity_delay < 0:
        raise ValueError("toxicity_delay must be nonnegative")
    return config.design, window, toxicity_delay


def _joint_probabilities(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if raw.shape != (4,):
        raise ValueError(
            "joint_probabilities must be [both, response-only, toxicity-only, neither]"
        )
    probabilities = finite(raw, "joint_probabilities")
    if np.any(probabilities < 0) or not np.isclose(
        np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("joint_probabilities must be nonnegative and sum to one")
    return _owned(probabilities / float(np.sum(probabilities)), np.float64)


def _draw_response_delays(
    rng: np.random.Generator,
    responses: NDArray[np.int8],
    window: float,
    timing: str,
) -> FloatArray:
    log_tail = np.log(0.05)
    if timing == "conditional_truncated_exponential":
        # Invert the exponential CDF conditional on being at or before the window.
        fractions = np.log1p(-0.95 * rng.random(responses.size)) / log_tail
    else:
        fractions = np.minimum(rng.exponential(size=responses.size) / -log_tail, 1.0)
    draws = window * fractions
    if np.any((responses == 1) & (fractions > 0) & (draws == 0)):
        raise ArithmeticError("positive response delay is below floating-point resolution")
    if np.any(~np.isfinite(draws)):
        raise ArithmeticError("response observation delays are not representable")
    return np.where(responses == 1, draws, window).astype(np.float64, copy=False)


def simulate_multc_trial(
    config: MultcSimulationConfig,
    joint_probabilities: ArrayLike,
    *,
    accrual_rate: float,
    seed: int,
) -> MultcCalendarTrial:
    """Generate paired binary outcomes/timing, then replay one Multc Lean trial.

    Joint category order is both endpoints, response only, toxicity only,
    neither. Durations are measured from first enrollment, which is time zero as
    required by the calendar replay, not from study opening. Later accrual-open
    gaps are exponential with mean ``1/accrual_rate``. Pauses freeze those gaps
    through the replay's documented convention.
    """
    design, window, toxicity_delay = _validate_config(config)
    probabilities = _joint_probabilities(joint_probabilities)
    seed_number = _seed_value(seed)
    rate = scalar(accrual_rate, "accrual_rate")
    mean_gap = 1.0 / rate if rate > 0 else float("inf")
    if not np.isfinite(mean_gap) or mean_gap <= 0:
        raise ValueError("accrual_rate must be positive with a finite mean gap")
    streams = np.random.SeedSequence(seed_number).spawn(3)
    outcome_rng, accrual_rng, response_time_rng = (
        np.random.default_rng(child) for child in streams
    )
    n = design.max_subjects
    categories = outcome_rng.choice(4, size=n, p=probabilities)
    outcomes = np.zeros((n, 2), dtype=np.int8)
    outcomes[categories == 0] = (1, 1)
    outcomes[categories == 1] = (1, 0)
    outcomes[categories == 2] = (0, 1)
    intervals = accrual_rng.exponential(scale=mean_gap, size=max(0, n - 1))
    if np.any(~np.isfinite(intervals)):
        raise ArithmeticError("accrual intervals are not representable")
    response_delays = _draw_response_delays(
        response_time_rng,
        outcomes[:, 0],
        window,
        config.response_timing,
    )
    toxicity_delays = np.full(n, toxicity_delay, dtype=np.float64)
    return run_multc_calendar_trial(
        design,
        outcomes,
        intervals,
        response_delays,
        toxicity_delays,
    )


def simulate_multc(
    config: MultcSimulationConfig,
    joint_probabilities: ArrayLike,
    *,
    accrual_rate: float,
    trials: int,
    seed: int,
    max_total_work: int = 500_000_000,
    max_total_storage_bytes: int = 512_000_000,
) -> MultcSimulationResult:
    """Run bounded serial Multc Lean simulations and summarize duration/decisions."""
    design, _, _ = _validate_config(config)
    probabilities = _joint_probabilities(joint_probabilities)
    rate = scalar(accrual_rate, "accrual_rate")
    mean_gap = 1.0 / rate if rate > 0 else float("inf")
    if not np.isfinite(mean_gap) or mean_gap <= 0:
        raise ValueError("accrual_rate must be positive with a finite mean gap")
    seed_number = _seed_value(seed)
    trial_count = _positive_integer(trials, "trials", _MAX_TRIALS)
    work_limit = _positive_integer(max_total_work, "max_total_work", _MAX_TOTAL_WORK)
    storage_limit = _positive_integer(
        max_total_storage_bytes, "max_total_storage_bytes", _MAX_TOTAL_STORAGE
    )
    n = design.max_subjects
    per_trial_work = 10 * n * n + 4 * n + len(design.looks)
    total_work = trial_count * per_trial_work
    look_count = len(design.looks)
    # Include generated inputs, replay copies/output arrays, and the replay's
    # worst-case history: one record/look plus up to two pending-endpoint events
    # per patient. Python record/container overhead is conservatively budgeted.
    per_trial_peak = n * 256 + (look_count + 2 * n) * 768 + 32_768
    # Seeds, six per-trial scalar vectors, quantile scratch and output ownership.
    summary_bytes = trial_count * 80 + (n + 1) * 8 + 4 * 8 * trial_count
    peak_storage = summary_bytes + per_trial_peak
    if total_work > work_limit:
        raise ValueError("max_total_work is below the aggregate worst-case estimate")
    if peak_storage > storage_limit:
        raise ValueError("max_total_storage_bytes is below the aggregate peak estimate")

    seed_sequence = np.random.SeedSequence(seed_number)
    seeds = np.empty(trial_count, dtype=np.uint64)
    enrollment = np.zeros(n + 1, dtype=np.int64)
    response_totals = np.empty(trial_count, dtype=np.float64)
    toxicity_totals = np.empty(trial_count, dtype=np.float64)
    durations = np.empty(trial_count, dtype=np.float64)
    accrual_durations = np.empty(trial_count, dtype=np.float64)
    paused_durations = np.empty(trial_count, dtype=np.float64)
    last_followup = np.empty(trial_count, dtype=np.float64)
    decision_counts: Counter[str] = Counter()

    for index in range(trial_count):
        child = seed_sequence.spawn(1)[0]
        trial_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        seeds[index] = trial_seed
        trial = simulate_multc_trial(
            config,
            probabilities,
            accrual_rate=rate,
            seed=trial_seed,
        )
        enrollment[trial.enrolled] += 1
        response_totals[index] = trial.responses
        toxicity_totals[index] = trial.toxicities
        durations[index] = trial.duration
        accrual_durations[index] = trial.accrual_end_time
        paused_durations[index] = trial.paused_duration
        last_followup[index] = trial.last_followup_time
        decision_counts[trial.decision] += 1
        del trial

    decisions = tuple(sorted(decision_counts))
    counts = np.asarray([decision_counts[name] for name in decisions], dtype=np.int64)
    decision_probability = counts.astype(np.float64) / trial_count
    decision_mcse = (
        np.full(counts.shape, np.nan)
        if trial_count == 1
        else np.sqrt(decision_probability * (1 - decision_probability) / trial_count)
    )
    duration_mean, duration_error = _scaled_mean_mcse(durations)
    accrual_mean, accrual_error = _scaled_mean_mcse(accrual_durations)
    paused_mean, paused_error = _scaled_mean_mcse(paused_durations)
    followup_mean, followup_error = _scaled_mean_mcse(last_followup)
    enrollment_probability = enrollment.astype(np.float64) / trial_count
    enrollment_values = np.arange(n + 1, dtype=np.float64)
    mean_enrolled = float(np.dot(enrollment, enrollment_values) / trial_count)
    if trial_count == 1:
        enrollment_error = float("nan")
    else:
        enrollment_variance = float(
            np.dot(enrollment, (enrollment_values - mean_enrolled) ** 2) / (trial_count - 1)
        )
        enrollment_error = float(np.sqrt(enrollment_variance / trial_count))
    return MultcSimulationResult(
        trial_count,
        probabilities,
        _owned(seeds, np.uint64),
        mean_enrolled,
        enrollment_error,
        _owned(enrollment_probability),
        *_bounded_mean_mcse(response_totals),
        *_bounded_mean_mcse(toxicity_totals),
        duration_mean,
        duration_error,
        _owned(np.quantile(durations, [0.025, 0.975], method="linear")),
        accrual_mean,
        accrual_error,
        _owned(np.quantile(accrual_durations, [0.025, 0.975], method="linear")),
        paused_mean,
        paused_error,
        followup_mean,
        followup_error,
        decisions,
        _owned(counts, np.int64),
        _owned(decision_probability),
        _owned(decision_mcse),
        per_trial_work,
        total_work,
        peak_storage,
    )
