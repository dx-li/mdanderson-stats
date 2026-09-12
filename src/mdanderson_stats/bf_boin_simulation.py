"""Calendar-time BF-BOIN simulation with asynchronous assessment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .bf_boin import BFBOINDesign
from .boin import _owned


@dataclass(frozen=True)
class BFBOINSimulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    assigned: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_assigned: FloatArray
    assigned_history: tuple[NDArray[np.int64], ...]
    arrival_history: tuple[FloatArray, ...]
    assessment_history: tuple[FloatArray, ...]
    dlt_history: tuple[NDArray[np.bool_], ...]
    response_history: tuple[NDArray[np.bool_], ...]
    backfill_history: tuple[NDArray[np.bool_], ...]
    escalation_end: FloatArray
    trial_duration: FloatArray

    @property
    def final_assessments(self) -> tuple[FloatArray, ...]:
        """Compatibility alias for per-patient DLT assessment times."""
        return self.assessment_history


@dataclass
class _PatientRecord:
    dose: int
    arrival: float
    dlt_assessment: float
    response_assessment: float
    dlt: bool
    response: bool
    dlt_seen: bool = False
    response_seen: bool = False
    backfill: bool = False


def _weibull_endpoint(probability: float, window: float) -> tuple[float, float]:
    """Calibrate Weibull F(window)=p and F(window/2)=p/2."""
    if probability == 0:
        return 1.0, np.inf
    if probability >= 1:
        return 1.0, 0.0
    a = -np.log1p(-probability)
    b = -np.log1p(-probability / 2)
    shape = np.log(a / b) / np.log(2.0)
    scale = np.exp(np.log(window) - np.log(a) / shape)
    return float(shape), float(scale)


def _positive(value: float, name: str) -> float:
    result = float(np.asarray(value, dtype=float))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def simulate_bf_boin(
    design: BFBOINDesign,
    true_toxicity: ArrayLike,
    true_response: ArrayLike,
    *,
    cohorts: ArrayLike = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    accrual_rate: float = 1.0,
    dlt_window: float = 1.0,
    arrival_distribution: str = "uniform",
    rng: int | np.random.Generator | None = None,
) -> BFBOINSimulation:
    """Run primary-mode BF-BOIN with separate DLT and response observation.

    Responses are sampled at enrollment and observed at arrival plus the DLT
    window. DLT times are calibrated Weibull endpoints and observed at the
    earlier event time or window. Arrivals use one persistent renewal schedule;
    the first arrival is at time zero.
    """
    if not isinstance(design, BFBOINDesign):
        raise ValueError("design must be a BFBOINDesign")
    toxicity, response = (
        finite(true_toxicity, "true_toxicity"),
        finite(true_response, "true_response"),
    )
    if (
        toxicity.ndim != 1
        or not 2 <= toxicity.size <= 100
        or response.shape != toxicity.shape
        or np.any((toxicity < 0) | (toxicity > 1))
        or np.any((response < 0) | (response > 1))
    ):
        raise ValueError("true_toxicity and true_response must match with probabilities in [0,1]")
    sizes = count([cohort_size, trials, start_dose], "simulation sizes")
    if sizes.shape != (3,) or np.any(sizes < 1):
        raise ValueError("cohort_size, trials and start_dose must be positive integers")
    size, repetitions, start = map(int, sizes)
    if repetitions > 1_000_000 or start > toxicity.size:
        raise ValueError("require at most 1000000 trials and a valid start_dose")
    cohort_counts = np.broadcast_to(count(cohorts, "cohorts"), (repetitions,))
    if np.any(cohort_counts < 1):
        raise ValueError("cohorts must contain positive integers")
    rate, window = _positive(accrual_rate, "accrual_rate"), _positive(dlt_window, "dlt_window")
    if arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
    bound = repetitions * (int(np.max(cohort_counts)) * size + toxicity.size * design.n_cap)
    if bound > 100_000:
        raise ValueError("requested trials and retained patient records exceed 100000")
    generator = np.random.default_rng(rng)
    endpoints = tuple(_weibull_endpoint(float(p), window) for p in toxicity)
    patients = np.zeros((repetitions, toxicity.size), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    assigned = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    reasons, dose_histories, arrival_histories = [], [], []
    assessment_histories, dlt_histories, response_histories, backfill_histories = [], [], [], []
    escalation_ends, durations = np.zeros(repetitions), np.zeros(repetitions)

    for trial in range(repetitions):
        evaluated = np.zeros(toxicity.size, dtype=np.int64)
        observed_dlt = np.zeros_like(evaluated)
        activity, excluded = np.zeros(toxicity.size, bool), np.zeros(toxicity.size, bool)
        # dose, arrival, dlt assessment, response assessment, dlt, response,
        # dlt_seen, response_seen, backfill
        records: list[_PatientRecord] = []
        next_arrival, clock, dose = 0.0, 0.0, start
        reason = "max_cohorts"
        arrival_steps = 0
        arrival_limit = 4 * (int(np.max(cohort_counts)) * size + toxicity.size * design.n_cap + 1)

        def observe(until: float) -> None:
            for r in records:
                if not r.dlt_seen and r.dlt_assessment <= until:
                    evaluated[r.dose] += 1
                    observed_dlt[r.dose] += int(r.dlt)
                    r.dlt_seen = True
                if not r.response_seen and r.response_assessment <= until:
                    activity[r.dose] |= r.response
                    r.response_seen = True

        def enroll(j: int, when: float, backfill: bool) -> int:
            shape, scale = endpoints[j]
            dlt_time = (
                np.inf
                if np.isinf(scale)
                else (window / 2 if scale == 0 else scale * generator.weibull(shape))
            )
            records.append(
                _PatientRecord(
                    j,
                    when,
                    when + min(dlt_time, window),
                    when + window,
                    dlt_time <= window,
                    bool(generator.random() < response[j]),
                    backfill=backfill,
                )
            )
            assigned[trial, j] += 1
            return len(records) - 1

        for _ in range(int(cohort_counts[trial])):
            current: list[int] = []
            while len(current) < size:
                clock = next_arrival
                observe(clock)
                current.append(enroll(dose - 1, clock, False))
                next_arrival = clock + (
                    generator.uniform(0, 2 / rate)
                    if arrival_distribution == "uniform"
                    else generator.exponential(1 / rate)
                )
            while not all(records[i].dlt_seen for i in current):
                next_dlt = min(
                    records[i].dlt_assessment for i in current if not records[i].dlt_seen
                )
                if next_arrival < next_dlt:
                    arrival_steps += 1
                    if arrival_steps > arrival_limit:
                        next_arrival = next_dlt
                        continue
                    clock = next_arrival
                    observe(clock)
                    eligibility = design.backfill_eligibility(
                        evaluated,
                        observed_dlt,
                        assigned[trial],
                        dose,
                        response_observed=activity,
                        eliminated=excluded,
                    )
                    if eligibility.dose is not None:
                        enroll(eligibility.dose - 1, clock, True)
                        next_arrival = clock + (
                            generator.uniform(0, 2 / rate)
                            if arrival_distribution == "uniform"
                            else generator.exponential(1 / rate)
                        )
                    else:
                        next_arrival = next_dlt
                else:
                    clock = next_dlt
                    observe(clock)
            decision = design.next_dose(
                evaluated,
                observed_dlt,
                assigned[trial],
                dose,
                backfilled=np.array(
                    [any(r.dose == j and r.backfill for r in records) for j in range(toxicity.size)]
                ),
                eliminated=excluded,
                response_observed=activity,
            )
            excluded = np.array(decision.eliminated, copy=True)
            if decision.next_dose is None:
                reason = decision.action
                break
            dose = decision.next_dose
        escalation_ends[trial] = clock
        observe(np.inf)
        patients[trial], toxicities[trial] = evaluated, observed_dlt
        durations[trial] = max((r.response_assessment for r in records), default=clock)
        selection = design.select_mtd(evaluated, observed_dlt, eliminated=excluded)
        selected[trial] = 0 if selection.dose is None else selection.dose
        reasons.append(reason)
        dose_histories.append(_owned(np.asarray([r.dose + 1 for r in records], dtype=np.int64)))
        arrival_histories.append(_owned(np.asarray([r.arrival for r in records])))
        assessment_histories.append(_owned(np.asarray([r.dlt_assessment for r in records])))
        dlt_histories.append(_owned(np.asarray([r.dlt for r in records])))
        response_histories.append(_owned(np.asarray([r.response for r in records])))
        backfill_histories.append(_owned(np.asarray([r.backfill for r in records])))
    frequency = np.bincount(selected, minlength=toxicity.size + 1) / repetitions
    return BFBOINSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(assigned),
        _owned(selected),
        tuple(reasons),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(0)),
        _owned(toxicities.mean(0)),
        _owned(assigned.mean(0)),
        tuple(dose_histories),
        tuple(arrival_histories),
        tuple(assessment_histories),
        tuple(dlt_histories),
        tuple(response_histories),
        tuple(backfill_histories),
        _owned(escalation_ends),
        _owned(durations),
    )
