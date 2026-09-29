"""Complete-outcome operating characteristics for ordinary BOIN combinations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import _owned
from .boin_combination import BOINCombDesign

Pair = tuple[int, int]


@dataclass(frozen=True)
class BOINCombinationSimulation:
    """Trial-level counts and replicated single-MTD operating characteristics."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    stop_reason: tuple[str, ...]
    titration_patients: NDArray[np.int64]
    titration_endpoint: NDArray[np.int64]
    titration_end_reason: tuple[str, ...]


def _pair(value: object, name: str, rows: int, columns: int) -> Pair:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != 2:
        raise ValueError(f"{name} must contain two one-based dose levels")
    try:
        first, second = (float(x) for x in array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two one-based dose levels") from exc
    if not np.isfinite(first + second) or first != int(first) or second != int(second):
        raise ValueError(f"{name} must contain two one-based dose levels")
    result = (int(first), int(second))
    if not (1 <= result[0] <= rows and 1 <= result[1] <= columns):
        raise ValueError(f"{name} must identify a dose pair on the supplied grid")
    return result


def simulate_boin_combination(
    design: BOINCombDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 20,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: Sequence[int] = (1, 1),
    titration: bool = False,
    rng: int | np.random.Generator | None = None,
) -> BOINCombinationSimulation:
    """Simulate BOIN combination trials, optionally using CRAN's titration prelude.

    With ``titration=True`` and cohorts larger than one, the trial starts with
    one patient at each visited cell. It moves one row or column upward with
    equal probability when both moves are available, and stops titrating at
    the first DLT or the upper-right cell. The first ordinary cohort tops the
    titration endpoint up to ``cohort_size``; ``cohorts`` still counts ordinary
    cohorts. Cohort size one disables titration, matching the CRAN simulator.
    """
    if not isinstance(design, BOINCombDesign):
        raise ValueError("design must be a BOINCombDesign")
    raw_probability = np.asarray(true_toxicity)
    if np.iscomplexobj(raw_probability):
        raise ValueError("true_toxicity must be real-valued")
    probability = finite(raw_probability, "true_toxicity")
    if (
        probability.ndim != 2
        or not all(2 <= size <= 100 for size in probability.shape)
        or probability.shape[0] > probability.shape[1]
    ):
        raise ValueError(
            "true_toxicity must be a 2..100 rows-by-columns matrix with rows <= columns"
        )
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("true_toxicity probabilities must be in [0,1]")
    settings = count([cohorts, cohort_size, trials], "simulation settings")
    if settings.shape != (3,) or np.any(settings < 1):
        raise ValueError("cohorts, cohort_size and trials must be positive integers")
    ncohort, size, repetitions = (int(value) for value in settings)
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    use_titration = bool(titration) and size > 1
    start = _pair(start_dose, "start_dose", *probability.shape)
    extra_staircase = (
        probability.shape[0] - start[0] + probability.shape[1] - start[1] if use_titration else 0
    )
    if ncohort * size + extra_staircase > 1000:
        raise ValueError("the trial may enroll at most 1000 patients")
    if repetitions > 1_000_000 or repetitions * probability.size > 2_000_000:
        raise ValueError("require at most 1000000 trials and 2000000 trial-dose cells")
    estimated_bytes = repetitions * (64 * probability.size + 768)
    if estimated_bytes > 128 * 1024 * 1024:
        raise ValueError("simulation retained and working arrays exceed the 128 MiB budget")
    generator = np.random.default_rng(rng)
    shape = probability.shape
    patients = np.zeros((repetitions, *shape), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    eliminated = np.zeros_like(patients, dtype=bool)
    selected = np.zeros((repetitions, 2), dtype=np.int64)
    reasons = np.full(repetitions, "max_cohorts", dtype="U32")
    current = [start] * repetitions
    titration_patients = np.zeros_like(patients)
    titration_endpoint = np.zeros((repetitions, 2), dtype=np.int64)
    default_titration_reason = (
        "cohort_size_one" if bool(titration) and size == 1 else "not_requested"
    )
    titration_reasons = np.full(repetitions, default_titration_reason, dtype="U16")
    if use_titration:
        for trial_index in range(repetitions):
            dose = start
            while True:
                i, j = dose[0] - 1, dose[1] - 1
                event = int(generator.binomial(1, probability[i, j]))
                patients[trial_index, i, j] += 1
                toxicities[trial_index, i, j] += event
                titration_patients[trial_index, i, j] += 1
                titration_endpoint[trial_index] = dose
                if event:
                    titration_reasons[trial_index] = "first_dlt"
                    break
                if i == shape[0] - 1 and j == shape[1] - 1:
                    titration_reasons[trial_index] = "upper_right"
                    break
                if i < shape[0] - 1 and j < shape[1] - 1:
                    if generator.integers(2) == 0:
                        dose = (dose[0] + 1, dose[1])
                    else:
                        dose = (dose[0], dose[1] + 1)
                elif i == shape[0] - 1:
                    dose = (dose[0], dose[1] + 1)
                else:
                    dose = (dose[0] + 1, dose[1])
            current[trial_index] = dose
    active = np.ones(repetitions, dtype=bool)
    for cohort_index in range(ncohort):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        coordinates = np.asarray([current[index] for index in rows], dtype=np.int64) - 1
        event_probability = probability[coordinates[:, 0], coordinates[:, 1]]
        if use_titration and cohort_index == 0:
            enrolled = size - patients[rows, coordinates[:, 0], coordinates[:, 1]]
            events = generator.binomial(enrolled, event_probability)
            patients[rows, coordinates[:, 0], coordinates[:, 1]] += enrolled
        else:
            events = generator.binomial(size, event_probability)
            patients[rows, coordinates[:, 0], coordinates[:, 1]] += size
        toxicities[rows, coordinates[:, 0], coordinates[:, 1]] += events
        for trial_index_value in rows:
            trial_index = int(trial_index_value)
            current_pair = current[trial_index]
            i, j = current_pair[0] - 1, current_pair[1] - 1
            current_n = int(patients[trial_index, i, j])
            current_y = int(toxicities[trial_index, i, j])
            if design.early_stop_patients is not None and current_n >= design.early_stop_patients:
                state, _ = design._state(
                    patients[trial_index], toxicities[trial_index], eliminated[trial_index]
                )
                if not state[0, 0]:
                    boundary = design.boundary_table(current_n)
                    index = current_n - 1
                    at_top = i == shape[0] - 1 and j == shape[1] - 1
                    blocked_up = i == shape[0] - 1 or state[i + 1, j]
                    blocked_right = j == shape[1] - 1 or state[i, j + 1]
                    converged = (
                        current_y > boundary.escalate_max[index]
                        or at_top
                        or (i == shape[0] - 1 and blocked_right)
                        or (j == shape[1] - 1 and blocked_up)
                        or (i < shape[0] - 1 and j < shape[1] - 1 and blocked_up and blocked_right)
                    ) and (current_y < boundary.deescalate_min[index] or (i == 0 and j == 0))
                    if converged:
                        reasons[trial_index] = "stop_precision"
                        active[trial_index] = False
                        eliminated[trial_index] = state
                        continue
            decision = design.next_dose(
                patients[trial_index],
                toxicities[trial_index],
                current[trial_index],
                eliminated=eliminated[trial_index],
                rng=generator,
                source_simulation=True,
            )
            if decision.next_dose is None:
                reasons[trial_index] = decision.action
                active[trial_index] = False
            else:
                current[trial_index] = decision.next_dose
            eliminated[trial_index] = decision.eliminated
    for final_index in range(repetitions):
        result = design.select_mtd(
            patients[final_index],
            toxicities[final_index],
            eliminated=eliminated[final_index],
            round_selection=False,
        )
        if result.dose is not None:
            selected[final_index] = result.dose
    frequency = np.zeros((shape[0] + 1, shape[1] + 1), dtype=float)
    np.add.at(frequency, (selected[:, 0], selected[:, 1]), 1)
    frequency /= repetitions
    return BOINCombinationSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(eliminated),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(axis=0)),
        _owned(toxicities.mean(axis=0)),
        tuple(str(reason) for reason in reasons),
        _owned(titration_patients),
        _owned(titration_endpoint),
        tuple(str(reason) for reason in titration_reasons),
    )
