"""Calendar replay for the patient-by-patient Dose Schedule Finder."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log1p

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from ._cdflib import _freeze
from .dose_schedule import (
    DoseSchedulePatient,
    _numeric,
    _positive_parameters,
    dose_schedule_cumulative_hazard,
    validate_schedules,
)
from .dose_schedule_decision import DoseScheduleDecision, dose_schedule_decision
from .dose_schedule_fit import DoseScheduleFit, fit_dose_schedule
from .dose_schedule_prior import DoseSchedulePrior

_MAX_PATIENTS = 200
_MAX_TOTAL_EVALUATIONS = 2_000_000
_MAX_TOTAL_WORK = 50_000_000
_MAX_PER_FIT_EVALUATIONS = 2_000_000
_MAX_PER_FIT_WORK = 50_000_000
_EVENT_ROOT_MAXITER = 200
_EVENT_ROOT_MAX_EVALUATIONS = _EVENT_ROOT_MAXITER + 4


def _count(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


def _vector(value: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    if isinstance(value, (list, tuple)) and len(value) > size:
        raise ValueError(f"{name} must contain exactly {size} values")
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector of length {size}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str, *, open_upper: bool = False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite probability")
    result = float(raw)
    if not isfinite(result) or result < 0 or result > 1 or (open_upper and result == 1):
        end = "[0,1)" if open_upper else "[0,1]"
        raise ValueError(f"{name} must lie in {end}")
    return result


def _seeds(value: ArrayLike, name: str, size: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and len(value) > size:
        raise ValueError(f"{name} must contain exactly {size} integer seeds")
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer vector of length {size}")
    seeds = tuple(int(item) for item in raw)
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"{name} must contain nonnegative seeds")
    return seeds


def _uniforms(value: ArrayLike, size: int) -> NDArray[np.float64]:
    raw = _vector(value, "event_uniforms", size)
    if np.any((raw <= 0) | (raw >= 1)):
        raise ValueError("event_uniforms must lie strictly between zero and one")
    return raw


def _hazard_at(
    elapsed: float,
    administration_times: NDArray[np.float64],
    area: float,
    peak: float,
    tail: float,
    threshold: float,
) -> float:
    total = 0.0
    cap = 2.0 * threshold
    for administration in administration_times:
        value = float(
            dose_schedule_cumulative_hazard(elapsed - float(administration), area, peak, tail)
        )
        if value >= cap - total:
            return cap
        total += value
    return total


def _event_time(
    uniform: float,
    administration_times: NDArray[np.float64],
    area: float,
    peak: float,
    tail: float,
    horizon: float,
) -> float:
    return _event_time_counted(uniform, administration_times, area, peak, tail, horizon)[0]


def _event_time_counted(
    uniform: float,
    administration_times: NDArray[np.float64],
    area: float,
    peak: float,
    tail: float,
    horizon: float,
) -> tuple[float, int]:
    threshold = -log1p(-uniform)
    if not isfinite(threshold) or threshold <= 0:
        raise ArithmeticError("event hazard threshold is not representable")
    maximum = _hazard_at(horizon, administration_times, area, peak, tail, threshold)
    if maximum < threshold:
        return float("inf"), 1

    evaluations = 1

    def residual(time: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return _hazard_at(time, administration_times, area, peak, tail, threshold) / threshold - 1.0

    try:
        event_time = brentq(
            residual,
            0.0,
            horizon,
            xtol=float(np.nextafter(0.0, 1.0)),
            rtol=4.0 * np.finfo(float).eps,
            maxiter=_EVENT_ROOT_MAXITER,
        )
    except (RuntimeError, ValueError) as exc:
        raise ArithmeticError(
            "event-time inversion did not converge within its bounded iterations"
        ) from exc
    if not isfinite(event_time) or event_time <= 0 or event_time > horizon:
        raise ArithmeticError("event-time inversion returned an invalid time")
    if abs(residual(event_time)) > 64.0 * np.finfo(float).eps:
        raise ArithmeticError("event-time inversion did not meet its hazard residual tolerance")
    return float(event_time), evaluations


def _observed_patient(
    arrival: float,
    current_time: float,
    event_time: float,
    horizon: float,
    schedule: NDArray[np.float64],
    dose_index: int,
) -> DoseSchedulePatient:
    elapsed = max(0.0, min(current_time - arrival, horizon))
    event = event_time <= elapsed
    observed_time = min(elapsed, event_time) if event else elapsed
    if event:
        actual = schedule[schedule < observed_time]
    else:
        actual = schedule[schedule <= observed_time]
    return DoseSchedulePatient(
        observed_time,
        event,
        actual,
        np.full(actual.size, dose_index, dtype=np.int64),
    )


def _max_diagnostic(summary: object, field: str) -> float:
    values = np.asarray(getattr(summary, field))
    if np.any(np.isinf(values)):
        return float("inf")
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


@dataclass(frozen=True)
class DoseScheduleTrialPatient:
    """One assigned patient and the complete final-follow-up history."""

    index: int
    arrival_time: float
    dose_index: int
    schedule_index: int
    event_uniform: float
    observed: DoseSchedulePatient


@dataclass(frozen=True)
class DoseScheduleTrialStep:
    """An initial assignment or an as-of-arrival posterior decision."""

    time: float
    patients_enrolled: int
    action: str
    pair: tuple[int, int] | None
    reason: str
    decision: DoseScheduleDecision | None
    fit_evaluations: int
    fit_work_units: int
    max_risk_split_rhat: float | None
    max_risk_mcse: float | None


@dataclass(frozen=True)
class DoseScheduleTrial:
    """Replay output with patient outcomes and only compact interim records.

    Tied planned arrivals are processed sequentially in input order. If an
    interim analysis finds no regimen that meets the safety and allocation
    rules, enrollment terminates permanently; follow-up and a final posterior
    fit still occur, but the final decision remains a stop. This stop policy
    is explicit Python replay bookkeeping because the source does not specify
    a post-stop recommendation.
    """

    patients: tuple[DoseScheduleTrialPatient, ...]
    steps: tuple[DoseScheduleTrialStep, ...]
    stop_reason: str
    stop_time: float
    final_time: float
    duration: float
    event_uniforms: NDArray[np.float64]
    sampler_seeds: tuple[int, ...]
    final_fit: DoseScheduleFit
    final_decision: DoseScheduleDecision
    total_likelihood_evaluations: int
    total_work_units: int


def run_dose_schedule_trial(
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
    rng: np.random.Generator | None = None,
    sampler_rng: np.random.Generator | None = None,
    event_uniforms: ArrayLike | None = None,
    sampler_seeds: ArrayLike | None = None,
    escalation_rule: str = "coordinate_max",
    draws: int,
    warmup: int,
    chains: int,
    max_evaluations_per_fit: int = 200_000,
    max_work_per_fit: int = 20_000_000,
    max_total_evaluations: int = 2_000_000,
    max_total_work: int = 50_000_000,
) -> DoseScheduleTrial:
    """Replay a bounded patient-by-patient dose/schedule trial.

    At each planned arrival after the first, current patients contribute only
    their as-of-arrival right-censored histories. A new patient receives the
    selected constant-dose candidate schedule; treatment stops at toxicity,
    while event-free patients continue to the horizon. The final posterior
    analysis waits until every enrolled patient has an event or full-horizon
    follow-up. Toxicity wins an exact tie with a planned administration, so
    that administration and later ones are omitted.

    The event-only model does not implement the paper example's delayed
    low-grade-to-DLT classification. Existing explicit no-skip policies are
    reused; their history conventions are Python choices. ``rng`` generates
    the bounded planned event-uniform tape and ``sampler_rng`` generates the
    bounded planned fit-seed tape. Supplied tapes replace their corresponding
    stream; returned tapes include unused suffixes so an early-stopped trial
    can be replayed directly. NumPy does not claim native software RNG parity.
    """
    patient_limit = _count(max_patients, "max_patients", 1, _MAX_PATIENTS)
    if not isinstance(prior, DoseSchedulePrior):
        raise ValueError("prior must be a DoseSchedulePrior")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    if sampler_rng is not None and not isinstance(sampler_rng, np.random.Generator):
        raise ValueError("sampler_rng must be a NumPy Generator")
    if rng is not None and rng is sampler_rng:
        raise ValueError("outcome and sampler RNGs must be distinct streams")
    if event_uniforms is None and rng is None:
        raise ValueError("provide rng or event_uniforms")
    if sampler_seeds is None and sampler_rng is None:
        raise ValueError("provide sampler_rng or sampler_seeds")

    arrivals = _vector(arrival_times, "arrival_times", patient_limit)
    if np.any(arrivals < 0) or np.any(np.diff(arrivals) < 0):
        raise ValueError("arrival_times must be nonnegative and nondecreasing")
    raw_horizon = np.asarray(horizon)
    if raw_horizon.ndim != 0 or raw_horizon.dtype.kind not in "iuf":
        raise ValueError("horizon must be a finite positive scalar")
    horizon_value = float(raw_horizon)
    if not isfinite(horizon_value) or horizon_value <= 0:
        raise ValueError("horizon must be finite and positive")
    with np.errstate(over="ignore"):
        followup_endpoints = arrivals + horizon_value
    if not np.all(np.isfinite(followup_endpoints)):
        raise ValueError("arrival time plus horizon exceeds floating-point range")
    if np.any(followup_endpoints <= arrivals):
        raise ValueError("horizon is too small to advance at one or more arrival times")
    candidates = validate_schedules(schedules, horizon_value)
    if prior.dose_count > 20 or not candidates:
        raise ValueError("prior and schedules must define a nonempty supported grid")
    area = _numeric(truth_area, "truth_area", max_size=20)
    peak = _numeric(truth_peak, "truth_peak", max_size=20)
    tail = _numeric(truth_tail, "truth_tail", max_size=20)
    if area.shape != (prior.dose_count,) or peak.shape != area.shape or tail.shape != area.shape:
        raise ValueError("truth area/peak/tail vectors must match the prior dose grid")
    _positive_parameters(area, peak, tail)

    if event_uniforms is not None:
        uniform_tape = _uniforms(event_uniforms, patient_limit)
    else:
        uniform_tape = None
    if sampler_seeds is not None:
        seed_tape = _seeds(sampler_seeds, "sampler_seeds", patient_limit)
    else:
        seed_tape = None

    draw_count = _count(draws, "draws", 8, 100_000)
    warmup_count = _count(warmup, "warmup", 0, 100_000)
    chain_count = _count(chains, "chains", 2, 8)
    per_fit_evaluations = _count(
        max_evaluations_per_fit, "max_evaluations_per_fit", 1, _MAX_PER_FIT_EVALUATIONS
    )
    per_fit_work = _count(max_work_per_fit, "max_work_per_fit", 1, _MAX_PER_FIT_WORK)
    total_evaluations_limit = _count(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    total_work_limit = _count(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    safety_limit = _probability(toxicity_limit, "toxicity_limit")
    upper_cutoff = _probability(upper_probability, "upper_probability", open_upper=True)
    if upper_cutoff == 0:
        raise ValueError("upper_probability must lie in (0,1)")
    target_value = _probability(target, "target")
    if escalation_rule not in ("coordinate_max", "observed_pair"):
        raise ValueError("escalation_rule must be coordinate_max or observed_pair")
    start = (0, 0)

    minimum_evaluations = chain_count * (1 + warmup_count + draw_count)
    risk_work = chain_count * draw_count * prior.dose_count * sum(row.size for row in candidates)
    maximum_schedule_length = max(row.size for row in candidates)
    fit_count = patient_limit
    patient_fit_sum = patient_limit * (patient_limit + 1) // 2
    preflight_evaluations = fit_count * minimum_evaluations
    parameter_count = 3 * prior.dose_count
    retained_cells = (
        chain_count
        * draw_count
        * (parameter_count + 3 * prior.dose_count + prior.dose_count * len(candidates) + 1)
    )
    if retained_cells > 2_000_000:
        raise ValueError("a retained dose-schedule fit exceeds two million cells")
    maximum_administrations = patient_limit * maximum_schedule_length
    minimum_per_fit_work = minimum_evaluations * maximum_administrations + risk_work
    if minimum_evaluations > per_fit_evaluations:
        raise ValueError("max_evaluations_per_fit is below the minimum chain workload")
    if minimum_per_fit_work > per_fit_work:
        raise ValueError("max_work_per_fit is below the minimum fit workload")
    preflight_work = (
        minimum_evaluations * maximum_schedule_length * patient_fit_sum
        + fit_count * risk_work
        + _EVENT_ROOT_MAX_EVALUATIONS * maximum_schedule_length * patient_limit
    )
    if preflight_evaluations > total_evaluations_limit:
        raise ValueError("max_total_evaluations is below the trial's minimum refit budget")
    if preflight_work > total_work_limit:
        raise ValueError("max_total_work is below the trial's conservative minimum budget")

    if uniform_tape is None:
        assert rng is not None
        uniform_tape = np.asarray(rng.random(patient_limit), dtype=np.float64)
        uniform_tape[uniform_tape == 0.0] = np.nextafter(0.0, 1.0)
        if np.any(uniform_tape >= 1.0):
            raise ArithmeticError("event RNG produced an out-of-range uniform value")
    if seed_tape is None:
        assert sampler_rng is not None
        seed_tape = tuple(
            int(item)
            for item in sampler_rng.integers(
                0, np.iinfo(np.uint64).max, size=patient_limit, dtype=np.uint64
            )
        )

    assigned_doses: list[int] = []
    assigned_schedules: list[int] = []
    event_times: list[float] = []
    used_seeds: list[int] = []
    steps: list[DoseScheduleTrialStep] = []
    work_units = 0
    likelihood_evaluations = 0

    def get_uniform(index: int) -> float:
        return float(uniform_tape[index])

    def get_seed(index: int) -> int:
        return seed_tape[index]

    def add_patient(index: int, pair: tuple[int, int]) -> None:
        nonlocal work_units
        dose_index, schedule_index = pair
        uniform = get_uniform(index)
        event_time, evaluations = _event_time_counted(
            uniform,
            candidates[schedule_index],
            float(area[dose_index]),
            float(peak[dose_index]),
            float(tail[dose_index]),
            horizon_value,
        )
        generated_work = evaluations * candidates[schedule_index].size
        if work_units + generated_work > total_work_limit:
            raise RuntimeError("event-time workload exceeded max_total_work")
        work_units += generated_work
        assigned_doses.append(dose_index)
        assigned_schedules.append(schedule_index)
        event_times.append(event_time)

    add_patient(0, start)
    steps.append(
        DoseScheduleTrialStep(
            float(arrivals[0]),
            1,
            "start",
            start,
            "first patient follows the configured lowest/start regimen",
            None,
            0,
            0,
            None,
            None,
        )
    )
    stop_reason = "max_patients"
    stop_time = float(arrivals[0])
    next_index = 1

    while next_index < patient_limit:
        decision_time = float(arrivals[next_index])
        current_patients = [
            _observed_patient(
                float(arrivals[i]),
                decision_time,
                event_times[i],
                horizon_value,
                candidates[assigned_schedules[i]],
                assigned_doses[i],
            )
            for i in range(len(assigned_doses))
        ]
        seed = get_seed(len(used_seeds))
        remaining_eval = total_evaluations_limit - likelihood_evaluations
        remaining_work = total_work_limit - work_units
        minimum_current_work = (
            minimum_evaluations * sum(patient.dose_indices.size for patient in current_patients)
            + risk_work
        )
        if remaining_eval < minimum_evaluations or remaining_work < minimum_current_work:
            raise RuntimeError("trial exhausted its total refit budget before a required analysis")
        fit = fit_dose_schedule(
            current_patients,
            prior,
            candidates,
            horizon_value,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            rng=np.random.default_rng(seed),
            max_evaluations=min(per_fit_evaluations, remaining_eval),
            max_work=min(per_fit_work, remaining_work),
        )
        used_seeds.append(seed)
        likelihood_evaluations += fit.likelihood_evaluations
        work_units += fit.work_units
        treated = np.zeros((prior.dose_count, len(candidates)), dtype=np.int64)
        np.add.at(treated, (assigned_doses, assigned_schedules), 1)
        decision = dose_schedule_decision(
            fit,
            treated,
            toxicity_limit=safety_limit,
            upper_probability=upper_cutoff,
            target=target_value,
            final=False,
            escalation_rule=escalation_rule,
        )
        steps.append(
            DoseScheduleTrialStep(
                decision_time,
                len(assigned_doses),
                decision.action,
                decision.pair,
                decision.reason,
                decision,
                fit.likelihood_evaluations,
                fit.work_units,
                _max_diagnostic(fit.risk_summary, "split_rhat"),
                _max_diagnostic(fit.risk_summary, "batch_mean_mcse"),
            )
        )
        stop_time = decision_time
        if decision.pair is None:
            stop_reason = (
                "no_safe_regimen" if not np.any(decision.acceptable) else "no_eligible_regimen"
            )
            del fit
            break
        del fit
        add_patient(next_index, decision.pair)
        next_index += 1
        if next_index == patient_limit:
            stop_time = float(arrivals[next_index - 1])

    complete_patients = tuple(
        DoseScheduleTrialPatient(
            index=i,
            arrival_time=float(arrivals[i]),
            dose_index=assigned_doses[i],
            schedule_index=assigned_schedules[i],
            event_uniform=float(uniform_tape[i]),
            observed=_observed_patient(
                0.0,
                horizon_value,
                event_times[i],
                horizon_value,
                candidates[assigned_schedules[i]],
                assigned_doses[i],
            ),
        )
        for i in range(len(assigned_doses))
    )
    final_patients = [item.observed for item in complete_patients]
    seed = get_seed(len(used_seeds))
    remaining_eval = total_evaluations_limit - likelihood_evaluations
    remaining_work = total_work_limit - work_units
    final_administration_count = sum(patient.dose_indices.size for patient in final_patients)
    final_minimum_work = minimum_evaluations * final_administration_count + risk_work
    if remaining_eval < minimum_evaluations or remaining_work < final_minimum_work:
        raise RuntimeError("trial exhausted its total refit budget before final analysis")
    final_fit = fit_dose_schedule(
        final_patients,
        prior,
        candidates,
        horizon_value,
        draws=draw_count,
        warmup=warmup_count,
        chains=chain_count,
        rng=np.random.default_rng(seed),
        max_evaluations=min(per_fit_evaluations, remaining_eval),
        max_work=min(per_fit_work, remaining_work),
    )
    used_seeds.append(seed)
    likelihood_evaluations += final_fit.likelihood_evaluations
    work_units += final_fit.work_units
    treated = np.zeros((prior.dose_count, len(candidates)), dtype=np.int64)
    np.add.at(treated, (assigned_doses, assigned_schedules), 1)
    final_decision = dose_schedule_decision(
        final_fit,
        treated,
        toxicity_limit=safety_limit,
        upper_probability=upper_cutoff,
        target=target_value,
        final=True,
        escalation_rule=escalation_rule,
    )
    if stop_reason in ("no_safe_regimen", "no_eligible_regimen"):
        final_decision = DoseScheduleDecision(
            "stop",
            None,
            final_decision.mean_risk,
            final_decision.overdose_probability,
            final_decision.acceptable,
            np.frombuffer(
                np.zeros(final_decision.acceptable.shape, dtype=np.bool_).tobytes(),
                dtype=np.bool_,
            ).reshape(final_decision.acceptable.shape),
            None,
            f"trial already terminated permanently after interim {stop_reason.replace('_', ' ')}",
        )
    final_time = max(
        [stop_time]
        + [patient.arrival_time + patient.observed.time for patient in complete_patients]
    )
    return DoseScheduleTrial(
        complete_patients,
        tuple(steps),
        stop_reason,
        stop_time,
        final_time,
        final_time - float(arrivals[0]),
        _freeze(uniform_tape),
        tuple(seed_tape),
        final_fit,
        final_decision,
        likelihood_evaluations,
        work_units,
    )
