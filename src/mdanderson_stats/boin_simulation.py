"""Complete-outcome, optionally accelerated-titration operating characteristics for local BOIN."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import BOINDesign, _owned


@dataclass(frozen=True)
class BOINSimulation:
    """Trial-level counts and one-based MTDs (zero denotes no selection).

    Selection probabilities and their Monte Carlo standard errors have entries
    [no selection, dose 1, ..., dose J]. Count arrays have shape (trials, doses).
    """

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    safety_stop_probability: float
    precision_stop_probability: float
    titration_patients: NDArray[np.int64]
    titration_moderate_toxicities: NDArray[np.int64]
    titration_end_reason: tuple[str, ...]


def simulate_boin(
    design: BOINDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: ArrayLike = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    titration: bool = False,
    titration_cap: int | None = None,
    moderate_toxicity: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> BOINSimulation:
    """Simulate independent Bernoulli DLTs with all outcomes known before decisions.

    Reuses the conduct and selection APIs, including sticky exclusions and optional
    precision stopping. Optional titration uses single-patient escalation until
    a DLT, the second grade-2 event, or a dose cap. Grade-2 probabilities are
    unconditional and mutually exclusive with DLTs; omitted means zero. Delayed
    outcomes are not included. Use compare_boin_three_plus_three for comparisons.
    Cohorts may be a scalar or one count per trial. Streams differ from R seeds.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    probability = finite(true_toxicity, "true_toxicity")
    if (
        probability.ndim != 1
        or not 2 <= probability.size <= 100
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("true_toxicity must contain 2..100 probabilities in [0,1]")
    sizes = count([cohort_size, trials, start_dose], "simulation sizes")
    if sizes.shape != (3,) or np.any(sizes < 1):
        raise ValueError("cohorts, cohort_size, trials and start_dose must be positive integers")
    size, repetitions, start = map(int, sizes)
    if repetitions > 1_000_000:
        raise ValueError("require at most 1000000 trials")
    cohort_counts = np.broadcast_to(count(cohorts, "cohorts"), (repetitions,))
    if np.any(cohort_counts < 1):
        raise ValueError("cohorts must contain positive integers")
    if start > probability.size or np.any(cohort_counts * size > 100_000):
        raise ValueError("require a valid start dose, at most 100000 patients and 1000000 trials")
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    cap_value = probability.size if titration_cap is None else titration_cap
    cap_count = count(cap_value, "titration_cap")
    if cap_count.ndim != 0 or not start <= cap_count <= probability.size:
        raise ValueError("titration_cap must be between start_dose and the highest dose")
    cap = int(cap_count)
    moderate = np.zeros_like(probability)
    if moderate_toxicity is not None:
        moderate = finite(moderate_toxicity, "moderate_toxicity")
        if moderate.shape != probability.shape or np.any(
            (moderate < 0) | (moderate + probability > 1)
        ):
            raise ValueError("moderate_toxicity must match doses and satisfy 0 <= grade2 <= 1-DLT")
    if not titration and (titration_cap is not None or moderate_toxicity is not None):
        raise ValueError("titration options require titration=True")
    generator = np.random.default_rng(rng)
    patients = np.zeros((repetitions, probability.size), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    reasons = []
    titration_counts = np.zeros(repetitions, dtype=np.int64)
    moderate_counts = np.zeros_like(patients)
    titration_reasons = []
    for trial in range(repetitions):
        maximum = int(cohort_counts[trial]) * size
        dose = start
        excluded = np.zeros(probability.size, dtype=bool)
        reason = "max_patients"
        enrolled = 0
        next_size = size
        titration_reason = "disabled"
        if titration and size > 1 and start < probability.size:
            while enrolled < maximum:
                j = dose - 1
                outcome = generator.random()
                dlt = outcome < probability[j]
                grade2 = not dlt and outcome < probability[j] + moderate[j]
                patients[trial, j] += 1
                toxicities[trial, j] += dlt
                moderate_counts[trial, j] += grade2
                enrolled += 1
                titration_counts[trial] += 1
                if dlt or moderate_counts[trial].sum() >= 2 or dose == probability.size:
                    titration_reason = (
                        "DLT"
                        if dlt
                        else "grade2"
                        if moderate_counts[trial].sum() >= 2
                        else "highest_dose"
                    )
                    next_size = size - 1
                    break
                if dose == cap:
                    dose += 1
                    titration_reason = "dose_cap"
                    break
                dose += 1
            else:
                titration_reason = "max_patients"
        titration_reasons.append(titration_reason)
        while enrolled < maximum:
            j = dose - 1
            batch = min(next_size, maximum - enrolled)
            patients[trial, j] += batch
            toxicities[trial, j] += generator.binomial(batch, probability[j])
            enrolled += batch
            next_size = size
            decision = design.next_dose(
                patients[trial], toxicities[trial], dose, eliminated=excluded
            )
            excluded = decision.eliminated
            if decision.next_dose is None:
                reason = decision.action
                break
            dose = decision.next_dose
        selection = design.select_mtd(patients[trial], toxicities[trial], eliminated=excluded)
        selected[trial] = 0 if selection.dose is None else selection.dose
        reasons.append(reason)
    frequency = np.bincount(selected, minlength=probability.size + 1) / repetitions
    return BOINSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(selected),
        tuple(reasons),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(axis=0)),
        _owned(toxicities.mean(axis=0)),
        reasons.count("stop_safety") / repetitions,
        reasons.count("stop_precision") / repetitions,
        _owned(titration_counts),
        _owned(moderate_counts),
        tuple(titration_reasons),
    )
