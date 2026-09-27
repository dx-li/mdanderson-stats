"""Bounded serial operating characteristics for BOIN waterfall trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .boin import _owned
from .boin_combination import BOINCombDesign
from .boin_waterfall_trial import (
    BOINWaterfallTrial,
    DoseCombination,
    WaterfallCohort,
    _budgets,
    _initial_space,
    _integer,
    _matrix,
    run_boin_waterfall_trial,
)

_MAX_TRIALS = 100_000
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_COHORT_RECORDS = 200_000
_MAX_OUTCOME_DRAWS = 20_000_000


@dataclass(frozen=True)
class WaterfallTrialHistory:
    """Compact executed subtrial spaces and patient-assignment trace."""

    dose_spaces: tuple[tuple[DoseCombination, ...], ...]
    start_positions: tuple[int, ...]
    cohort_budgets: tuple[int, ...]
    cohorts: tuple[WaterfallCohort, ...]
    row_candidates: tuple[DoseCombination | None, ...]
    source_contour: tuple[DoseCombination | None, ...]
    selected_contour: tuple[DoseCombination | None, ...]
    continuity_blocked: NDArray[np.bool_]
    stop_reason: str


@dataclass(frozen=True)
class BOINWaterfallSimulation:
    """Per-trial actual counts and row-specific contour selection frequencies.

    ``selected_columns`` is one-based by row; zero denotes no recommendation.
    Selection probabilities contain actual dose columns only, so each row may
    sum below one; the separate no-selection probabilities account for the rest.
    """

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_columns: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_probability: FloatArray
    no_selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    histories: tuple[WaterfallTrialHistory, ...]
    stop_reason: tuple[str, ...]
    total_patients: NDArray[np.int64]
    total_toxicities: NDArray[np.int64]


def _frozen_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _frozen_bool(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    result = np.array(values, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def simulate_boin_waterfall(
    design: BOINCombDesign,
    true_toxicity: ArrayLike,
    cohort_budgets: ArrayLike,
    *,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    early_stop_patients: int = 12,
    bound_mtd: bool = False,
    rng: int | np.random.Generator | None = None,
) -> BOINWaterfallSimulation:
    """Run independent waterfall trials serially and retain compact histories.

    Each trial receives a fresh sequential uniform tape; trial conduct consumes
    it in assigned cohort order. Titration is not simulated in this release.
    """
    if not isinstance(design, BOINCombDesign):
        raise ValueError("design must be a BOINCombDesign")
    probability = _matrix(true_toxicity, "true_toxicity", probability=True)
    rows, columns = probability.shape
    size = _integer(cohort_size, "cohort_size", 1, 1000)
    budgets, maximum = _budgets(cohort_budgets, rows, size)
    repetitions = _integer(trials, "trials", 1, _MAX_TRIALS)
    start = _integer(start_dose, "start_dose", 1, len(_initial_space(rows, columns)))
    early = _integer(early_stop_patients, "early_stop_patients", 1, 1000)
    if not isinstance(bound_mtd, (bool, np.bool_)):
        raise ValueError("bound_mtd must be boolean")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raw_rng = np.asarray(rng)
        if (
            raw_rng.ndim != 0
            or raw_rng.dtype.kind not in "iu"
            or isinstance(rng, (bool, np.bool_))
            or int(raw_rng) < 0
        ):
            raise ValueError("rng must be a nonnegative integer seed, Generator or None")
    cells = repetitions * (3 * rows * columns + 3 * rows + 2) + 4 * rows * columns + 2 * rows
    cohort_records = repetitions * sum(budgets)
    outcome_draws = repetitions * maximum
    if (
        cells > _MAX_OUTPUT_CELLS
        or cohort_records > _MAX_COHORT_RECORDS
        or outcome_draws > _MAX_OUTCOME_DRAWS
    ):
        raise ValueError("simulation exceeds bounded output, history or outcome-draw limits")

    generator = np.random.default_rng(rng)
    shape = (repetitions, rows, columns)
    patients = np.zeros(shape, dtype=np.int64)
    toxicities = np.zeros(shape, dtype=np.int64)
    eliminated = np.zeros(shape, dtype=bool)
    selected = np.zeros((repetitions, rows), dtype=np.int64)
    totals_n = np.zeros(repetitions, dtype=np.int64)
    totals_y = np.zeros(repetitions, dtype=np.int64)
    reasons: list[str] = []
    histories: list[WaterfallTrialHistory] = []
    for trial_index in range(repetitions):
        uniforms = generator.random(maximum)
        result: BOINWaterfallTrial = run_boin_waterfall_trial(
            design,
            probability,
            budgets,
            uniforms,
            cohort_size=size,
            start_dose=start,
            early_stop_patients=early,
            bound_mtd=bound_mtd,
        )
        patients[trial_index] = result.patients
        toxicities[trial_index] = result.toxicities
        eliminated[trial_index] = result.eliminated
        totals_n[trial_index] = result.total_patients
        totals_y[trial_index] = result.total_toxicities
        for row, pair in enumerate(result.selected_contour):
            if pair is not None:
                selected[trial_index, row] = pair[1]
        reasons.append(result.stop_reason)
        histories.append(
            WaterfallTrialHistory(
                tuple(subtrial.dose_space for subtrial in result.subtrials),
                tuple(subtrial.start_position for subtrial in result.subtrials),
                tuple(subtrial.cohort_budget for subtrial in result.subtrials),
                result.cohort_history,
                result.row_candidates,
                result.source_contour,
                result.selected_contour,
                result.continuity_blocked,
                result.stop_reason,
            )
        )

    frequencies = np.zeros((rows, columns), dtype=float)
    no_selection = np.zeros(rows, dtype=float)
    for row in range(rows):
        valid = selected[:, row] >= 1
        no_selection[row] = 1.0 - float(np.mean(valid))
        if np.any(valid):
            frequencies[row] = (
                np.bincount(selected[valid, row] - 1, minlength=columns) / repetitions
            )
    return BOINWaterfallSimulation(
        _frozen_int(patients),
        _frozen_int(toxicities),
        _frozen_bool(eliminated),
        _frozen_int(selected),
        _owned(frequencies),
        _owned(np.sqrt(frequencies * (1 - frequencies) / repetitions)),
        _owned(no_selection),
        _owned(np.sqrt(no_selection * (1 - no_selection) / repetitions)),
        _owned(patients.mean(axis=0)),
        _owned(toxicities.mean(axis=0)),
        tuple(histories),
        tuple(reasons),
        _frozen_int(totals_n),
        _frozen_int(totals_y),
    )
