"""Bounded replay and serial simulation for the author global MTADF rule."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .mtadf import _scalar
from .mtadf_author_global import (
    MTADFAuthorGlobalFit,
    _author_admissible_count,
    _author_global_next_dose,
    mtadf_author_global_decision,
    mtadf_author_global_fit,
)

_MAX_DOSES = 20
_MAX_COHORTS = 1_000
_MAX_PATIENTS = 1_000
_MAX_TRIALS = 10_000
_MAX_TOTAL_FITS = 20_000
_MAX_TOTAL_OUTCOME_CELLS = 2_000_000
_MAX_TOTAL_SUMMARY_CELLS = 2_000_000


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, np.ndarray)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return result


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= _MAX_DOSES or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a scalar vector with 2..{_MAX_DOSES} entries")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded array, list, or tuple")
    if raw.ndim != 1 or not 2 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with 2..{_MAX_DOSES} entries")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return result


def _outcome_tape(value: ArrayLike, name: str, cohorts: int, cohort_size: int) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if len(value) != cohorts or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must have one scalar count per cohort")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional array")
    if raw.ndim != 1 or raw.size != cohorts or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have one real count per cohort")
    numeric = np.asarray(raw, dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > cohort_size)
        or np.any(numeric >= 2**53)
    ):
        raise ValueError(f"{name} entries must be integer counts in [0, cohort_size]")
    return _freeze_int(numeric)


def _tape_length(value: ArrayLike, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= _MAX_COHORTS:
            raise ValueError(f"{name} must contain 1..{_MAX_COHORTS} cohort counts")
        return int(value.size)
    if isinstance(value, (list, tuple)) and 1 <= len(value) <= _MAX_COHORTS:
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain scalar cohort counts")
        return len(value)
    raise ValueError(f"{name} must be a bounded one-dimensional cohort tape")


def _limits(toxicity_limit: float, safety_cutoff: float) -> tuple[float, float]:
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    return phi, cutoff


@dataclass(frozen=True, slots=True)
class MTADFAuthorGlobalTrialResult:
    """Compact observed-outcome tape and decision ledger for one trial."""

    subjects: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    assigned_dose: NDArray[np.int64]
    cohort_toxicities: NDArray[np.int64]
    cohort_responses: NDArray[np.int64]
    admissible_count_before: NDArray[np.int64]
    admissible_count_after: NDArray[np.int64]
    selected_dose: int
    final_fit: MTADFAuthorGlobalFit
    fit_count: int
    nonconverged_fit_count: int
    total_irls_iterations: int
    max_irls_iterations: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class MTADFAuthorGlobalSimulation:
    """Immutable operating-characteristic summaries from serial trials."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    assigned_dose_history: NDArray[np.int64]
    cohort_toxicity_history: NDArray[np.int64]
    cohort_response_history: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray
    fit_count: int
    nonconverged_fit_count: int
    total_irls_iterations: int
    max_irls_iterations: int
    seed: int | None
    stop_reason: tuple[str, ...]


def _run_trial(
    dose_count: int,
    cohort_count: int,
    cohort_size: int,
    draw_outcomes: Callable[[int, int], tuple[int, int]],
    toxicity_limit: float,
    safety_cutoff: float,
) -> MTADFAuthorGlobalTrialResult:
    """Run the shared conduct engine with an indexed observed-count source."""
    n = np.zeros(dose_count, dtype=np.int64)
    y = np.zeros(dose_count, dtype=np.int64)
    r = np.zeros(dose_count, dtype=np.int64)
    current = 0
    cap, _, _ = _author_admissible_count(
        n.astype(float), y.astype(float), toxicity_limit, safety_cutoff
    )
    assigned: list[int] = []
    tox_tape: list[int] = []
    response_tape: list[int] = []
    cap_before: list[int] = []
    cap_after: list[int] = []
    fits = 0
    nonconverged = 0
    total_iterations = 0
    last_fit: MTADFAuthorGlobalFit | None = None

    maximum_iterations = 0
    for cohort_index in range(cohort_count):
        observed = draw_outcomes(cohort_index, current)
        tox_count, response_count = observed
        if (
            isinstance(tox_count, (bool, np.bool_))
            or isinstance(response_count, (bool, np.bool_))
            or not isinstance(tox_count, (int, np.integer))
            or not isinstance(response_count, (int, np.integer))
            or not 0 <= int(tox_count) <= cohort_size
            or not 0 <= int(response_count) <= cohort_size
        ):
            raise ValueError("outcome source must return two integer cohort counts")
        assigned.append(current)
        tox_tape.append(int(tox_count))
        response_tape.append(int(response_count))
        cap_before.append(cap)
        n[current] += cohort_size
        y[current] += int(tox_count)
        r[current] += int(response_count)

        # The author global rule refits after every cohort, including when the
        # admissible prefix has only one dose.
        last_fit = mtadf_author_global_fit(n, r)
        fits += 1
        nonconverged += int(not last_fit.converged)
        total_iterations += last_fit.iterations
        maximum_iterations = max(maximum_iterations, last_fit.iterations)
        if not last_fit.converged:
            raise ArithmeticError("author global logistic fit did not converge")
        target = int(
            np.flatnonzero(last_fit.fitted_efficacy == np.max(last_fit.fitted_efficacy))[-1]
        )
        current = _author_global_next_dose(current, target, cap)
        cap, _, _ = _author_admissible_count(
            n.astype(float), y.astype(float), toxicity_limit, safety_cutoff
        )
        cap_after.append(cap)

    assert last_fit is not None
    final = mtadf_author_global_decision(
        n,
        y,
        r,
        final=True,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
        fit=last_fit,
    )
    return MTADFAuthorGlobalTrialResult(
        _freeze_int(n),
        _freeze_int(y),
        _freeze_int(r),
        _freeze_int(assigned),
        _freeze_int(tox_tape),
        _freeze_int(response_tape),
        _freeze_int(cap_before),
        _freeze_int(cap_after),
        final.dose,
        last_fit,
        fits,
        nonconverged,
        total_iterations,
        maximum_iterations,
        "maximum_enrollment",
    )


def replay_mtadf_author_global_trial(
    cohort_toxicities: ArrayLike,
    cohort_responses: ArrayLike,
    *,
    dose_count: int,
    cohort_size: int = 3,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
) -> MTADFAuthorGlobalTrialResult:
    """Replay observed per-cohort outcomes under the author global rule.

    The two vectors give toxicity and efficacy counts for the assigned dose at
    each review. They are not all-dose potential-outcome tables. The returned
    assignment vector makes the compact tape replayable and auditable.
    """
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    doses = _integer(dose_count, "dose_count", 2, _MAX_DOSES)
    phi, cutoff = _limits(toxicity_limit, safety_cutoff)
    cohorts = _tape_length(cohort_toxicities, "cohort_toxicities")
    if _tape_length(cohort_responses, "cohort_responses") != cohorts:
        raise ValueError("cohort outcome tapes must have matching lengths")
    if cohorts * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    tox = _outcome_tape(cohort_toxicities, "cohort_toxicities", cohorts, size)
    eff = _outcome_tape(cohort_responses, "cohort_responses", cohorts, size)

    def observed_counts(index: int, _dose: int) -> tuple[int, int]:
        return int(tox[index]), int(eff[index])

    return _run_trial(doses, cohorts, size, observed_counts, phi, cutoff)


def simulate_mtadf_author_global(
    true_toxicity: ArrayLike,
    true_efficacy: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 100,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    rng: int | np.integer | np.random.Generator | None = None,
    max_total_fits: int = _MAX_TOTAL_FITS,
    max_total_outcome_cells: int = _MAX_TOTAL_OUTCOME_CELLS,
    max_total_summary_cells: int = _MAX_TOTAL_SUMMARY_CELLS,
) -> MTADFAuthorGlobalSimulation:
    """Run bounded assigned-dose binomial trials and summarize selections.

    For each cohort the RNG draws toxicity then efficacy at the currently
    assigned dose only. The integer seed (if supplied) is retained exactly;
    a caller-owned Generator is accepted but its original seed is unknown.
    """
    tox = _probabilities(true_toxicity, "true_toxicity")
    eff = _probabilities(true_efficacy, "true_efficacy")
    if tox.shape != eff.shape:
        raise ValueError("true_toxicity and true_efficacy must have matching dose counts")
    cohort_count = _integer(cohorts, "cohorts", 1, _MAX_COHORTS)
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    if cohort_count * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    fit_limit = _integer(max_total_fits, "max_total_fits", 1, _MAX_TOTAL_FITS)
    outcome_limit = _integer(
        max_total_outcome_cells,
        "max_total_outcome_cells",
        1,
        _MAX_TOTAL_OUTCOME_CELLS,
    )
    summary_limit = _integer(
        max_total_summary_cells,
        "max_total_summary_cells",
        1,
        _MAX_TOTAL_SUMMARY_CELLS,
    )
    total_fits = trial_count * cohort_count
    outcome_cells = trial_count * cohort_count * 2
    summary_cells = trial_count * (tox.size * 3 + cohort_count * 3 + 1) + tox.size * 5
    if total_fits > fit_limit:
        raise ValueError("requested simulation exceeds max_total_fits")
    if outcome_cells > outcome_limit:
        raise ValueError("requested simulation exceeds max_total_outcome_cells")
    if summary_cells > summary_limit:
        raise ValueError("requested simulation exceeds max_total_summary_cells")
    phi, cutoff = _limits(toxicity_limit, safety_cutoff)
    seed: int | None
    if isinstance(rng, np.random.Generator):
        generator = rng
        seed = None
    elif rng is None:
        seed = None
        generator = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
        seed = int(rng)
        if not 0 <= seed < 2**64:
            raise ValueError("integer rng seed must be in [0, 2**64)")
        generator = np.random.default_rng(seed)
    else:
        raise ValueError("rng must be a uint64 integer seed, Generator, or None")

    n_by_trial = np.zeros((trial_count, tox.size), dtype=np.int64)
    y_by_trial = np.zeros_like(n_by_trial)
    r_by_trial = np.zeros_like(n_by_trial)
    selected = np.empty(trial_count, dtype=np.int64)
    assigned_history = np.empty((trial_count, cohort_count), dtype=np.int64)
    cohort_tox_history = np.empty_like(assigned_history)
    cohort_response_history = np.empty_like(assigned_history)
    nonconverged_count = 0
    total_iterations = 0
    maximum_iterations = 0
    reasons: list[str] = []
    for trial_index in range(trial_count):

        def draw(index: int, dose: int) -> tuple[int, int]:
            del index
            tox_count = int(generator.binomial(size, tox[dose]))
            response_count = int(generator.binomial(size, eff[dose]))
            return tox_count, response_count

        result = _run_trial(tox.size, cohort_count, size, draw, phi, cutoff)
        n_by_trial[trial_index] = result.subjects
        y_by_trial[trial_index] = result.toxicities
        r_by_trial[trial_index] = result.responses
        selected[trial_index] = result.selected_dose
        assigned_history[trial_index] = result.assigned_dose
        cohort_tox_history[trial_index] = result.cohort_toxicities
        cohort_response_history[trial_index] = result.cohort_responses
        nonconverged_count += result.nonconverged_fit_count
        total_iterations += result.total_irls_iterations
        maximum_iterations = max(maximum_iterations, result.max_irls_iterations)
        reasons.append(result.stop_reason)

    frequencies = np.bincount(selected, minlength=tox.size).astype(np.float64)
    selection = frequencies / trial_count
    selection_mcse = np.sqrt(selection * (1.0 - selection) / trial_count)
    return MTADFAuthorGlobalSimulation(
        _freeze_int(n_by_trial),
        _freeze_int(y_by_trial),
        _freeze_int(r_by_trial),
        _freeze_int(selected),
        _freeze_int(assigned_history),
        _freeze_int(cohort_tox_history),
        _freeze_int(cohort_response_history),
        _freeze(selection),
        _freeze(selection_mcse),
        _freeze(np.mean(n_by_trial, axis=0)),
        _freeze(np.mean(y_by_trial, axis=0)),
        _freeze(np.mean(r_by_trial, axis=0)),
        total_fits,
        nonconverged_count,
        total_iterations,
        maximum_iterations,
        seed,
        tuple(reasons),
    )
