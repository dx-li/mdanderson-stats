"""Complete-outcome, fixed-cohort operating characteristics for local BOIN."""

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


def simulate_boin(
    design: BOINDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> BOINSimulation:
    """Simulate independent Bernoulli DLTs with all outcomes known before decisions.

    Reuses the conduct and selection APIs, including sticky exclusions and optional
    stay-only precision stopping. No accelerated titration, delayed outcomes, or
    3+3 comparator is implied. Random streams differ from R even with equal seeds.
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
    sizes = count([cohorts, cohort_size, trials, start_dose], "simulation sizes")
    if sizes.shape != (4,) or np.any(sizes < 1):
        raise ValueError("cohorts, cohort_size, trials and start_dose must be positive integers")
    nc, size, repetitions, start = map(int, sizes)
    if start > probability.size or nc * size > 100_000 or repetitions > 1_000_000:
        raise ValueError("require a valid start dose, at most 100000 patients and 1000000 trials")
    generator = np.random.default_rng(rng)
    patients = np.zeros((repetitions, probability.size), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    reasons = []
    for trial in range(repetitions):
        dose = start
        excluded = np.zeros(probability.size, dtype=bool)
        reason = "max_patients"
        for _ in range(nc):
            j = dose - 1
            patients[trial, j] += size
            toxicities[trial, j] += generator.binomial(size, probability[j])
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
    )
