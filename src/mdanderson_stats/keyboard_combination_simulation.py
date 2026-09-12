"""Complete-outcome KeyboardComb trial simulation and operating characteristics.

The simulator enrolls complete cohorts, draws independent binary DLT outcomes,
delegates safety and neighboring-dose transitions to ``KeyboardCombDesign``,
and delegates final MTD selection to its matrix-isotonic selector. Results
retain trial-level counts, stopping reasons, selection probabilities, and
Monte Carlo standard errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import _owned
from .keyboard_combination import KeyboardCombDesign

Pair = tuple[int, int]


@dataclass(frozen=True)
class KeyboardCombinationSimulation:
    """Trial-level dose counts and replicated operating characteristics.

    Dose pairs use one-based labels.  ``(0, 0)`` denotes no selected MTD, so
    ``selection_probability`` and ``selection_mcse`` have shape
    ``(rows + 1, columns + 1)`` with the zero row/column reserved for no
    selection.  Count arrays have shape ``(trials, rows, columns)``.
    """

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    stop_reason: tuple[str, ...]


def _pair(value: object, name: str, rows: int, columns: int) -> Pair:
    a = np.asarray(value)
    if a.ndim != 1 or a.size != 2:
        raise ValueError(f"{name} must contain two one-based dose levels")
    try:
        first, second = (float(x) for x in a)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two one-based dose levels") from exc
    if not np.isfinite(first + second) or first != int(first) or second != int(second):
        raise ValueError(f"{name} must contain two one-based dose levels")
    result = (int(first), int(second))
    if not 1 <= result[0] <= rows or not 1 <= result[1] <= columns:
        raise ValueError(f"{name} must identify a dose pair on the supplied grid")
    return result


def _decision_pair(value: Pair | None, rows: int, columns: int) -> Pair | None:
    """Validate a one-based pair returned by the concrete core API."""

    if value is None:
        return None
    result = np.asarray(value)
    if result.ndim == 0 or result.size != 2:
        raise ArithmeticError("Keyboard combination decision did not return a dose pair")
    first, second = (float(x) for x in result.reshape(-1))
    if not np.isfinite(first + second) or first != int(first) or second != int(second):
        raise ArithmeticError("Keyboard combination decision did not return integer dose labels")
    pair = (int(first), int(second))
    if not 1 <= pair[0] <= rows or not 1 <= pair[1] <= columns:
        raise ArithmeticError("Keyboard combination decision returned an invalid dose pair")
    return pair


def simulate_keyboard_combination(
    design: KeyboardCombDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 20,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: Sequence[int] = (1, 1),
    rng: int | np.random.Generator | None = None,
) -> KeyboardCombinationSimulation:
    """Simulate independent binary DLT outcomes for KeyboardComb.

    Each active replicate receives one complete cohort at its current dose
    pair.  The supplied design then decides safety, stopping, and the next pair;
    all outcomes are observed before that transition.  The final MTD is chosen
    by ``design.select_mtd`` using all observed counts.  NumPy's seeded
    generator is intentionally used; this function does not promise R RNG
    parity.

    The design is evaluated for every replicate and cohort.  In particular,
    this preserves the core design's random tie resolution among equally good
    neighboring combinations.
    """

    p = finite(true_toxicity, "true_toxicity")
    if p.ndim != 2 or not all(2 <= d <= 100 for d in p.shape) or p.shape[0] > p.shape[1]:
        raise ValueError(
            "true_toxicity must be a 2..100 rows-by-columns matrix with rows <= columns"
        )
    if np.any((p < 0) | (p > 1)):
        raise ValueError("true_toxicity probabilities must be in [0,1]")
    settings = count([cohorts, cohort_size, trials], "simulation settings")
    if settings.shape != (3,) or np.any(settings < 1):
        raise ValueError("cohorts, cohort_size and trials must be positive integers")
    nc, size, repetitions = (int(x) for x in settings)
    if nc * size > 200:
        raise ValueError("the trial may enroll at most 200 patients")
    if repetitions > 1_000_000 or repetitions * p.size > 2_000_000:
        raise ValueError("require at most 1000000 trials and 2000000 trial-dose cells")
    start = _pair(start_dose, "start_dose", *p.shape)

    generator = np.random.default_rng(rng)
    shape = (p.shape[0], p.shape[1])
    patients = np.zeros((repetitions, *shape), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    eliminated = np.zeros_like(patients, dtype=bool)
    selected = np.zeros((repetitions, 2), dtype=np.int64)
    reasons = np.full(repetitions, "max_cohorts", dtype="U32")
    current = [start] * repetitions
    active = np.ones(repetitions, dtype=bool)
    for _ in range(nc):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        coords = np.asarray([current[i] for i in rows], dtype=np.int64) - 1
        probability = p[coords[:, 0], coords[:, 1]]
        events = generator.binomial(size, probability)
        patients[rows, coords[:, 0], coords[:, 1]] += size
        toxicities[rows, coords[:, 0], coords[:, 1]] += events
        for trial_index in rows:
            decision = design.next_dose(
                patients[trial_index],
                toxicities[trial_index],
                current[trial_index],
                eliminated=eliminated[trial_index],
                rng=generator,
            )
            next_pair = _decision_pair(decision.next_dose, *shape)
            updated = np.asarray(decision.eliminated, dtype=bool)
            if updated.shape != shape:
                raise ArithmeticError(
                    "Keyboard combination decision returned invalid elimination mask"
                )
            eliminated[trial_index] = updated
            if next_pair is None:
                reasons[trial_index] = decision.action
                active[trial_index] = False
            else:
                current[trial_index] = next_pair

    for trial in range(repetitions):
        result = design.select_mtd(patients[trial], toxicities[trial], eliminated=eliminated[trial])
        chosen = _decision_pair(result.dose, *shape)
        if chosen is not None:
            selected[trial] = chosen

    frequency = np.zeros((shape[0] + 1, shape[1] + 1), dtype=float)
    np.add.at(frequency, (selected[:, 0], selected[:, 1]), 1)
    frequency /= repetitions
    return KeyboardCombinationSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(eliminated),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(axis=0)),
        _owned(toxicities.mean(axis=0)),
        tuple(str(x) for x in reasons),
    )
