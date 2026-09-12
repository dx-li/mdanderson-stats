"""Calendar-time BF-BOIN simulation with asynchronous DLT assessments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .bf_boin import BFBOINDesign
from .boin import _owned


@dataclass(frozen=True)
class BFBOINSimulation:
    """Operating characteristics and auditable calendar records."""

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
    final_assessments: tuple[FloatArray, ...]
    escalation_end: FloatArray
    trial_duration: FloatArray


def _weibull_endpoint(probability: float, window: float) -> tuple[float, float]:
    """Return Weibull shape and scale with F(w)=p and F(w/2)=p/2."""
    if probability == 0:
        return 1.0, np.inf
    if probability >= 1:
        return 1.0, 0.0
    a = -np.log1p(-probability)
    b = -np.log1p(-probability / 2)
    shape = np.log(a / b) / np.log(2.0)
    scale = window / a ** (1.0 / shape)
    return float(shape), float(scale)


def _gap(generator: np.random.Generator, rate: float, distribution: str) -> float:
    if distribution == "uniform":
        return float(generator.uniform(0.0, 2.0 / rate))
    return float(generator.exponential(1.0 / rate))


def simulate_bf_boin(
    design: BFBOINDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: ArrayLike = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    accrual_rate: float = 1.0,
    dlt_window: float = 1.0,
    n_esc: int | None = None,
    max_patients: int | None = None,
    arrival_distribution: str = "uniform",
    rng: int | np.random.Generator | None = None,
) -> BFBOINSimulation:
    """Simulate asynchronous BF-BOIN escalation and backfill.

    Patients arrive under a uniform ``(0, 2/accrual_rate)`` gap law by
    default, or exponential gaps when ``arrival_distribution='exponential'``.
    DLT times use calibrated Weibull endpoints and assessment occurs at
    ``min(dlt_time, dlt_window)``. Expansion-after-escalation and titration
    modes are intentionally outside this API.
    """
    if not isinstance(design, BFBOINDesign):
        raise ValueError("design must be a BFBOINDesign")
    probability = finite(true_toxicity, "true_toxicity")
    if (
        probability.ndim != 1
        or not 2 <= probability.size <= 100
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("true_toxicity must contain 2..100 probabilities in [0,1]")
    sizes = count([cohort_size, trials, start_dose], "simulation sizes")
    if sizes.shape != (3,) or np.any(sizes < 1):
        raise ValueError("cohort_size, trials and start_dose must be positive integers")
    size, repetitions, start = map(int, sizes)
    if repetitions > 1_000_000 or start > probability.size:
        raise ValueError("require at most 1000000 trials and a valid start_dose")
    cohort_counts = np.broadcast_to(count(cohorts, "cohorts"), (repetitions,))
    if np.any(cohort_counts < 1):
        raise ValueError("cohorts must contain positive integers")
    rate, window = (
        scalar_positive(accrual_rate, "accrual_rate"),
        scalar_positive(dlt_window, "dlt_window"),
    )
    if arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
    escalation_cohorts = int(n_esc) if n_esc is not None else int(np.max(cohort_counts))
    if escalation_cohorts < 1 or escalation_cohorts > 100_000:
        raise ValueError("n_esc must be in [1,100000]")
    limit = int(max_patients) if max_patients is not None else escalation_cohorts * size * 10
    if limit < escalation_cohorts * size or limit > 100_000:
        raise ValueError("max_patients must cover escalation cohorts and be at most 100000")
    retained_bound = repetitions * (escalation_cohorts * size + probability.size * design.n_cap)
    if retained_bound > 100_000:
        raise ValueError("requested trials and retained BF-BOIN records exceed 100000")
    generator = np.random.default_rng(rng)
    shapes_scales = tuple(_weibull_endpoint(float(p), window) for p in probability)
    patients = np.zeros((repetitions, probability.size), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    assigned = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    reasons: list[str] = []
    histories: list[NDArray[np.int64]] = []
    assessments: list[FloatArray] = []
    escalation_ends = np.zeros(repetitions, dtype=float)
    trial_durations = np.zeros(repetitions, dtype=float)

    for trial in range(repetitions):
        target_cohorts = int(cohort_counts[trial]) if n_esc is None else escalation_cohorts
        records: list[tuple[int, float, float, bool]] = []
        all_assessment_times: list[float] = []
        evaluated = np.zeros(probability.size, dtype=np.int64)
        observed_dlt = np.zeros(probability.size, dtype=np.int64)
        activity = np.zeros(probability.size, dtype=bool)
        dose = start
        excluded = np.zeros(probability.size, dtype=bool)
        clock = 0.0
        reason = "max_cohorts"

        def observe(until: float) -> None:
            nonlocal records
            pending = []
            for j, arrival, assess, dlt in records:
                if assess <= until and assess >= 0:
                    if not activity[j]:
                        activity[j] = True
                    evaluated[j] += 1
                    observed_dlt[j] += int(dlt)
                else:
                    pending.append((j, arrival, assess, dlt))
            records = pending

        def enroll(j: int, when: float, backfill: bool = False) -> None:
            dlt_time = np.inf
            shape, scale = shapes_scales[j]
            if scale == 0:
                dlt_time = window / 2
            elif np.isfinite(scale):
                dlt_time = float(scale * generator.weibull(shape))
            assess = when + min(dlt_time, window)
            dlt = bool(dlt_time <= window)
            records.append((j, when, assess, dlt))
            all_assessment_times.append(assess)
            assigned[trial, j] += 1
            if not backfill:
                patients[trial, j] += 1

        for _ in range(target_cohorts):
            current_ids: list[tuple[int, float, float, bool]] = []
            for _patient in range(size):
                clock += _gap(generator, rate, arrival_distribution) if records else 0.0
                enroll(dose - 1, clock)
                current_ids.append(records[-1])
            cohort_done = False
            while not cohort_done:
                next_assessment = min((r[2] for r in records), default=clock)
                arrival = clock + _gap(generator, rate, arrival_distribution)
                if arrival < next_assessment and assigned[trial].sum() < limit:
                    clock = arrival
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
                        enroll(eligibility.dose - 1, clock, backfill=True)
                else:
                    clock = next_assessment
                    observe(clock)
                cohort_done = all(r not in records for r in current_ids)
                if assigned[trial].sum() >= limit and not cohort_done:
                    clock = max(clock, next_assessment)
                    observe(clock)
            patients[trial] = evaluated
            toxicities[trial] = observed_dlt
            decision = design.next_dose(
                evaluated,
                observed_dlt,
                assigned[trial],
                dose,
                backfilled=assigned[trial] > evaluated,
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
        trial_durations[trial] = max(all_assessment_times, default=clock)
        patients[trial] = evaluated
        toxicities[trial] = observed_dlt
        selection = design.select_mtd(evaluated, observed_dlt, eliminated=excluded)
        selected[trial] = 0 if selection.dose is None else selection.dose
        if reason == "max_cohorts" and assigned[trial].sum() >= limit:
            reason = "max_patients"
        reasons.append(reason)
        histories.append(_owned(assigned[trial]))
        assessments.append(_owned(np.asarray(all_assessment_times, dtype=float)))

    frequency = np.bincount(selected, minlength=probability.size + 1) / repetitions
    return BFBOINSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(assigned),
        _owned(selected),
        tuple(reasons),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(axis=0)),
        _owned(toxicities.mean(axis=0)),
        _owned(assigned.mean(axis=0)),
        tuple(histories),
        tuple(assessments),
        _owned(escalation_ends),
        _owned(trial_durations),
    )


def scalar_positive(value: float, name: str) -> float:
    result = float(np.asarray(value, dtype=float))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result
