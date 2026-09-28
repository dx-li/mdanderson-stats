"""Bounded aggregate operating characteristics for Dose Schedule Finder."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .dose_schedule import _numeric, _positive_parameters, validate_schedules
from .dose_schedule_prior import DoseSchedulePrior
from .dose_schedule_trial import (
    _EVENT_ROOT_MAX_EVALUATIONS,
    _MAX_PATIENTS,
    _MAX_PER_FIT_EVALUATIONS,
    _MAX_PER_FIT_WORK,
    _MAX_TOTAL_EVALUATIONS,
    _MAX_TOTAL_WORK,
    _count,
    _probability,
    _vector,
    run_dose_schedule_trial,
)

_MAX_TRIALS = 10_000
_MAX_RETAINED_CELLS = 2_000_000
_STOP_REASONS = ("max_patients", "no_safe_regimen", "no_eligible_regimen")


def _maximum(current: float | None, value: float | None) -> float | None:
    if value is None or isnan(value):
        return current
    return value if current is None else max(current, value)


def _mean_mcse(total: NDArray[np.float64], squares: NDArray[np.float64], count: int):
    mean = total / count
    if count < 2:
        return mean, np.full_like(mean, np.nan)
    variance = np.maximum((squares - total * total / count) / (count - 1), 0.0)
    return mean, np.sqrt(variance / count)


def _pooled_rate_mcse(
    events: NDArray[np.float64],
    event_squares: NDArray[np.float64],
    event_patient_cross: NDArray[np.float64],
    patients: NDArray[np.float64],
    patient_squares: NDArray[np.float64],
    trials: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rate = np.full_like(events, np.nan)
    np.divide(events, patients, out=rate, where=patients > 0)
    mcse = np.full_like(rate, np.nan)
    if trials < 2:
        return rate, mcse
    residual = event_squares - 2 * rate * event_patient_cross + rate * rate * patient_squares
    residual = np.maximum(residual, 0.0)
    variance = np.full_like(rate, np.nan)
    np.divide(
        trials * residual,
        (trials - 1) * patients * patients,
        out=variance,
        where=patients > 0,
    )
    np.sqrt(variance, out=mcse, where=patients > 0)
    return rate, mcse


def _maximum_summary(summary: object, field: str) -> float | None:
    values = np.asarray(getattr(summary, field))
    if np.any(np.isinf(values)):
        return float("inf")
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class DoseScheduleOperatingCharacteristics:
    """Aggregate trial selections, allocation, outcomes and Monte Carlo error.

    ``event_seeds[i]`` and ``sampler_seeds[i]`` replay replicate ``i`` by
    passing separate generators initialized from those values to
    :func:`run_dose_schedule_trial`. Toxicity rates pool fully observed patient
    histories within each dose/schedule cell; their MCSEs use trial-clustered
    ratio estimates. Parameter diagnostics describe each final fit, while risk
    diagnostics take maxima over all retained interim and final summaries.
    Infinite diagnostics are preserved; maxima with no defined values are
    ``None``. ``diagnostic_undefined_values`` counts missing scalar summaries.
    """

    trials: int
    dose_count: int
    schedule_count: int
    selected_count: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_count: int
    no_selection_probability: float
    no_selection_mcse: float
    stop_reasons: tuple[str, ...]
    stop_reason_count: NDArray[np.int64]
    stop_reason_probability: FloatArray
    stop_reason_mcse: FloatArray
    early_stop_count: int
    early_stop_probability: float
    early_stop_mcse: float
    mean_enrollment: float
    enrollment_mcse: float
    mean_duration: float
    duration_mcse: float
    mean_allocation: FloatArray
    allocation_mcse: FloatArray
    assigned_patients: NDArray[np.int64]
    observed_toxicities: NDArray[np.int64]
    toxicity_probability: FloatArray
    toxicity_mcse: FloatArray
    event_seeds: NDArray[np.uint64]
    sampler_seeds: NDArray[np.uint64]
    final_parameter_rhat_max: float | None
    final_parameter_mcse_max: float | None
    all_fit_risk_rhat_max: float | None
    all_fit_risk_mcse_max: float | None
    diagnostic_undefined_values: int
    likelihood_evaluations: int
    work_units: int


def simulate_dose_schedule_operating_characteristics(
    truth_area: ArrayLike,
    truth_peak: ArrayLike,
    truth_tail: ArrayLike,
    prior: DoseSchedulePrior,
    schedules: tuple[ArrayLike, ...] | list[ArrayLike],
    horizon: float,
    arrival_times: ArrayLike,
    *,
    max_patients: int,
    toxicity_limit: float,
    upper_probability: float,
    target: float,
    trials: int,
    rng: np.random.Generator,
    escalation_rule: str = "coordinate_max",
    draws: int,
    warmup: int,
    chains: int,
    max_evaluations_per_fit: int = 200_000,
    max_work_per_fit: int = 20_000_000,
    max_total_evaluations: int = _MAX_TOTAL_EVALUATIONS,
    max_total_work: int = _MAX_TOTAL_WORK,
) -> DoseScheduleOperatingCharacteristics:
    """Run serial operating-characteristic replicates of the calendar replay.

    Each replicate uses a replayable independent event/sampler seed pair and
    calls :func:`run_dose_schedule_trial` with shared remaining work budgets.
    All configuration and conservative minimum-work/storage checks complete
    before the supplied RNG is advanced. Full histories and posterior draws are
    discarded after each replicate. No delayed low-grade classification,
    within-patient adaptation, automatic calibration or native RNG parity is
    introduced.
    """
    if not isinstance(prior, DoseSchedulePrior):
        raise TypeError("prior must be a DoseSchedulePrior")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit NumPy Generator")
    trial_count = _count(trials, "trials", 1, _MAX_TRIALS)
    patient_count = _count(max_patients, "max_patients", 1, _MAX_PATIENTS)
    draw_count = _count(draws, "draws", 8, 100_000)
    warmup_count = _count(warmup, "warmup", 0, 100_000)
    chain_count = _count(chains, "chains", 2, 8)
    per_fit_evaluations = _count(
        max_evaluations_per_fit,
        "max_evaluations_per_fit",
        1,
        _MAX_PER_FIT_EVALUATIONS,
    )
    per_fit_work = _count(max_work_per_fit, "max_work_per_fit", 1, _MAX_PER_FIT_WORK)
    evaluation_limit = _count(
        max_total_evaluations,
        "max_total_evaluations",
        1,
        _MAX_TOTAL_EVALUATIONS,
    )
    work_limit = _count(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    safety_limit = _probability(toxicity_limit, "toxicity_limit")
    upper_cutoff = _probability(upper_probability, "upper_probability", open_upper=True)
    if upper_cutoff == 0:
        raise ValueError("upper_probability must lie in (0,1)")
    target_value = _probability(target, "target")
    if escalation_rule not in ("coordinate_max", "observed_pair"):
        raise ValueError("escalation_rule must be coordinate_max or observed_pair")

    if isinstance(schedules, (list, tuple)) and len(schedules) > 20:
        raise ValueError("at most 20 administration schedules are supported")
    raw_horizon = np.asarray(horizon)
    if raw_horizon.ndim != 0 or raw_horizon.dtype.kind not in "iuf":
        raise ValueError("horizon must be a finite positive scalar")
    horizon_value = float(raw_horizon)
    if not isfinite(horizon_value) or horizon_value <= 0:
        raise ValueError("horizon must be finite and positive")
    arrivals = _vector(arrival_times, "arrival_times", patient_count)
    if np.any(arrivals < 0) or np.any(arrivals[1:] < arrivals[:-1]):
        raise ValueError("arrival_times must be nonnegative and nondecreasing")
    with np.errstate(over="ignore"):
        followup_end = arrivals + horizon_value
    if not np.all(np.isfinite(followup_end)) or np.any(followup_end <= arrivals):
        raise ValueError("arrival plus horizon must be finite and advance calendar time")
    candidate_schedules = validate_schedules(schedules, horizon_value)
    dose_count = prior.dose_count
    schedule_count = len(candidate_schedules)
    if dose_count > 20 or not schedule_count:
        raise ValueError("prior and schedules must define a nonempty supported grid")
    area = _numeric(truth_area, "truth_area", max_size=20)
    peak = _numeric(truth_peak, "truth_peak", max_size=20)
    tail = _numeric(truth_tail, "truth_tail", max_size=20)
    if area.shape != (dose_count,) or peak.shape != area.shape or tail.shape != area.shape:
        raise ValueError("truth area, peak and tail must match the prior dose grid")
    _positive_parameters(area, peak, tail)

    minimum_evaluations = chain_count * (1 + warmup_count + draw_count)
    minimum_fit_work = minimum_evaluations * patient_count * max(
        row.size for row in candidate_schedules
    ) + chain_count * draw_count * dose_count * sum(row.size for row in candidate_schedules)
    if minimum_evaluations > per_fit_evaluations:
        raise ValueError("max_evaluations_per_fit is below the minimum chain workload")
    if minimum_fit_work > per_fit_work:
        raise ValueError("max_work_per_fit is below the minimum trial fit workload")
    per_trial_eval_preflight = patient_count * minimum_evaluations
    patient_fit_sum = patient_count * (patient_count + 1) // 2
    risk_work = chain_count * draw_count * dose_count * sum(row.size for row in candidate_schedules)
    maximum_schedule_length = max(row.size for row in candidate_schedules)
    per_trial_work_preflight = (
        minimum_evaluations * maximum_schedule_length * patient_fit_sum
        + patient_count * risk_work
        + _EVENT_ROOT_MAX_EVALUATIONS * maximum_schedule_length * patient_count
    )
    if trial_count * per_trial_eval_preflight > evaluation_limit:
        raise ValueError("total evaluation budget cannot cover each trial's preflight minimum")
    if trial_count * per_trial_work_preflight > work_limit:
        raise ValueError("total work budget cannot cover each trial's preflight minimum")

    grid_cells = dose_count * schedule_count
    retained_output = trial_count * (2 + 2 * grid_cells) + 12 * grid_cells + 3 * len(_STOP_REASONS)
    one_fit = (
        chain_count
        * draw_count
        * (3 * dose_count + 3 * dose_count + dose_count * schedule_count + 1)
    )
    one_trial_history = patient_count * (maximum_schedule_length + 8) + (5 * patient_count + 2) * (
        4 * grid_cells + 16
    )
    if retained_output + one_fit + one_trial_history > _MAX_RETAINED_CELLS:
        raise ValueError("aggregate outputs plus one trial's fit/history exceed two million cells")

    entropy = rng.integers(0, 2**32, size=8, dtype=np.uint32)
    children = np.random.SeedSequence([int(value) for value in entropy]).spawn(trial_count)
    event_seeds = np.empty(trial_count, dtype=np.uint64)
    sampler_seeds = np.empty(trial_count, dtype=np.uint64)
    for index, child in enumerate(children):
        event_child, sampler_child = child.spawn(2)
        event_seeds[index] = event_child.generate_state(1, dtype=np.uint64)[0]
        sampler_seeds[index] = sampler_child.generate_state(1, dtype=np.uint64)[0]

    grid = (dose_count, schedule_count)
    selected_count = np.zeros(grid, dtype=np.int64)
    reason_count = np.zeros(len(_STOP_REASONS), dtype=np.int64)
    allocation_sum = np.zeros(grid)
    allocation_squares = np.zeros(grid)
    patients_sum = np.zeros(grid)
    patients_squares = np.zeros(grid)
    toxicity_sum = np.zeros(grid)
    toxicity_squares = np.zeros(grid)
    toxicity_patient_cross = np.zeros(grid)
    enrollment_total = enrollment_squares = 0.0
    duration_scale = duration_mean_scaled = duration_m2_scaled = 0.0
    processed_trials = 0
    no_selection_count = early_stop_count = 0
    final_parameter_rhat_max: float | None = None
    final_parameter_mcse_max: float | None = None
    risk_rhat_max: float | None = None
    risk_mcse_max: float | None = None
    undefined_diagnostics = 0
    used_evaluations = used_work = 0

    for event_seed, sampler_seed in zip(event_seeds, sampler_seeds, strict=True):
        remaining_evaluations = evaluation_limit - used_evaluations
        remaining_work = work_limit - used_work
        trial = run_dose_schedule_trial(
            area,
            peak,
            tail,
            prior,
            candidate_schedules,
            horizon_value,
            arrivals,
            max_patients=patient_count,
            toxicity_limit=safety_limit,
            upper_probability=upper_cutoff,
            target=target_value,
            rng=np.random.default_rng(int(event_seed)),
            sampler_rng=np.random.default_rng(int(sampler_seed)),
            escalation_rule=escalation_rule,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            max_evaluations_per_fit=per_fit_evaluations,
            max_work_per_fit=per_fit_work,
            max_total_evaluations=remaining_evaluations,
            max_total_work=remaining_work,
        )
        used_evaluations += trial.total_likelihood_evaluations
        used_work += trial.total_work_units
        if used_evaluations > evaluation_limit or used_work > work_limit:
            raise RuntimeError("aggregate Dose Schedule Finder work exceeded its shared budget")

        enrollment = len(trial.patients)
        enrollment_total += enrollment
        enrollment_squares += enrollment * enrollment
        duration = float(trial.duration)
        processed_trials += 1
        if duration > duration_scale:
            rescale = duration_scale / duration if duration_scale > 0 else 0.0
            duration_mean_scaled *= rescale
            duration_m2_scaled *= rescale * rescale
            duration_scale = duration
        normalized_duration = duration / duration_scale if duration_scale > 0 else 0.0
        duration_delta = normalized_duration - duration_mean_scaled
        duration_mean_scaled += duration_delta / processed_trials
        duration_m2_scaled += duration_delta * (normalized_duration - duration_mean_scaled)
        reason_index = _STOP_REASONS.index(trial.stop_reason)
        reason_count[reason_index] += 1
        early_stop_count += int(trial.stop_reason != "max_patients")
        pair = trial.final_decision.pair
        if pair is None:
            no_selection_count += 1
        else:
            selected_count[pair] += 1

        for step in trial.steps:
            if step.action == "start":
                continue
            for value, key in (
                (step.max_risk_split_rhat, "risk_rhat"),
                (step.max_risk_mcse, "risk_mcse"),
            ):
                if value is None or np.isnan(value):
                    undefined_diagnostics += 1
                elif key == "risk_rhat":
                    risk_rhat_max = _maximum(risk_rhat_max, value)
                else:
                    risk_mcse_max = _maximum(risk_mcse_max, value)
        parameter_value = _maximum_summary(trial.final_fit.parameter_summary, "split_rhat")
        if parameter_value is None:
            undefined_diagnostics += 1
        final_parameter_rhat_max = _maximum(final_parameter_rhat_max, parameter_value)
        parameter_mcse_value = _maximum_summary(
            trial.final_fit.parameter_summary, "batch_mean_mcse"
        )
        if parameter_mcse_value is None:
            undefined_diagnostics += 1
        final_parameter_mcse_max = _maximum(final_parameter_mcse_max, parameter_mcse_value)
        final_risk_rhat = _maximum_summary(trial.final_fit.risk_summary, "split_rhat")
        final_risk_mcse = _maximum_summary(trial.final_fit.risk_summary, "batch_mean_mcse")
        if final_risk_rhat is None:
            undefined_diagnostics += 1
        else:
            risk_rhat_max = _maximum(risk_rhat_max, final_risk_rhat)
        if final_risk_mcse is None:
            undefined_diagnostics += 1
        else:
            risk_mcse_max = _maximum(risk_mcse_max, final_risk_mcse)

        allocation = np.zeros(grid, dtype=float)
        events = np.zeros(grid, dtype=float)
        for patient in trial.patients:
            cell = (patient.dose_index, patient.schedule_index)
            allocation[cell] += 1.0
            events[cell] += float(patient.observed.event)
        allocation_sum += allocation
        allocation_squares += allocation * allocation
        patients_sum += allocation
        patients_squares += allocation * allocation
        toxicity_sum += events
        toxicity_squares += events * events
        toxicity_patient_cross += events * allocation
        del trial

    selection_probability = selected_count / trial_count
    selection_mcse = np.sqrt(selection_probability * (1.0 - selection_probability) / trial_count)
    no_selection_probability = no_selection_count / trial_count
    no_selection_mcse = sqrt(
        no_selection_probability * (1.0 - no_selection_probability) / trial_count
    )
    reason_probability = reason_count / trial_count
    reason_mcse = np.sqrt(reason_probability * (1.0 - reason_probability) / trial_count)
    early_stop_probability = early_stop_count / trial_count
    early_stop_mcse = sqrt(early_stop_probability * (1.0 - early_stop_probability) / trial_count)
    mean_allocation, allocation_mcse = _mean_mcse(allocation_sum, allocation_squares, trial_count)
    toxicity_probability, toxicity_mcse = _pooled_rate_mcse(
        toxicity_sum,
        toxicity_squares,
        toxicity_patient_cross,
        patients_sum,
        patients_squares,
        trial_count,
    )
    mean_enrollment = enrollment_total / trial_count
    mean_duration = duration_scale * duration_mean_scaled
    if trial_count < 2:
        enrollment_mcse = duration_mcse = float("nan")
    else:
        enrollment_mcse = sqrt(
            max(
                (enrollment_squares - enrollment_total * enrollment_total / trial_count)
                / (trial_count - 1),
                0.0,
            )
            / trial_count
        )
        duration_mcse = duration_scale * sqrt(
            max(duration_m2_scaled, 0.0) / ((trial_count - 1) * trial_count)
        )
    return DoseScheduleOperatingCharacteristics(
        trial_count,
        dose_count,
        schedule_count,
        _readonly(selected_count, np.int64),
        _readonly(selection_probability),
        _readonly(selection_mcse),
        no_selection_count,
        no_selection_probability,
        no_selection_mcse,
        _STOP_REASONS,
        _readonly(reason_count, np.int64),
        _readonly(reason_probability),
        _readonly(reason_mcse),
        early_stop_count,
        early_stop_probability,
        early_stop_mcse,
        mean_enrollment,
        enrollment_mcse,
        mean_duration,
        duration_mcse,
        _readonly(mean_allocation),
        _readonly(allocation_mcse),
        _readonly(patients_sum, np.int64),
        _readonly(toxicity_sum, np.int64),
        _readonly(toxicity_probability),
        _readonly(toxicity_mcse),
        _readonly(event_seeds, np.uint64),
        _readonly(sampler_seeds, np.uint64),
        final_parameter_rhat_max,
        final_parameter_mcse_max,
        risk_rhat_max,
        risk_mcse_max,
        undefined_diagnostics,
        used_evaluations,
        used_work,
    )
