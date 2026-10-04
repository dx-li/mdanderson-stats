"""Complete-cohort replay and serial simulation for the rendered MTADF R rule.

This module preserves the source simulator's one-cohort-lagged safety cap.
It is separate from :mod:`mtadf_simulation`, whose paper-policy behavior is
deliberately different.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .mtadf import _scalar
from .mtadf_author import (
    MTADFAuthorDecision,
    _mtadf_author_admissible_dose_count,
    _mtadf_author_decision_with_count,
    mtadf_author_decision,
)
from .mtadf_simulation import MTADFSimulation

_MAX_DOSES = 20
_MAX_COHORTS = 1_000
_MAX_PATIENTS = 1_000
_MAX_TRIALS = 10_000
_MAX_TOTAL_DECISIONS = 1_000_000
_MAX_TOTAL_POTENTIAL_CELLS = 2_000_000


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_DOSES or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a scalar vector with 1..{_MAX_DOSES} entries")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded array, list, or tuple")
    if raw.ndim != 1 or not 1 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with 1..{_MAX_DOSES} entries")
    result: FloatArray = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return result


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


def _count_matrix(value: ArrayLike, name: str, cohort_size: int) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_COHORTS:
            raise ValueError(f"{name} must have 1..{_MAX_COHORTS} cohort rows")
        for row in value:
            if isinstance(row, np.ndarray):
                valid_row = row.ndim == 1 and row.size <= _MAX_DOSES
                if valid_row and any(not np.isscalar(cell) for cell in row):
                    raise ValueError(f"{name} must contain scalar count cells")
            elif isinstance(row, (list, tuple)):
                valid_row = len(row) <= _MAX_DOSES
                if valid_row and any(not np.isscalar(cell) for cell in row):
                    raise ValueError(f"{name} must contain scalar count cells")
            else:
                valid_row = False
            if not valid_row:
                raise ValueError(f"{name} rows must have at most {_MAX_DOSES} scalar entries")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded array, list, or tuple")
    if (
        raw.ndim != 2
        or not 1 <= raw.shape[0] <= _MAX_COHORTS
        or not 1 <= raw.shape[1] <= _MAX_DOSES
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError(f"{name} must be a cohorts-by-doses real matrix within supported limits")
    numeric: FloatArray = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > cohort_size)
        or np.any(numeric >= 2**53)
    ):
        raise ValueError(f"{name} entries must be integer cohort counts in [0, cohort_size]")
    return _freeze_int(numeric)


@dataclass(frozen=True)
class MTADFAuthorTrialResult:
    """Immutable result and ledger for one author-convention trial replay."""

    subjects: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    assigned_dose: NDArray[np.int64]
    admissible_count_before: NDArray[np.int64]
    admissible_count_after: NDArray[np.int64]
    selected_dose: int
    final_decision: MTADFAuthorDecision
    stop_reason: str


def replay_mtadf_author_trial(
    cohort_toxicities: ArrayLike,
    cohort_responses: ArrayLike,
    *,
    cohort_size: int = 3,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
) -> MTADFAuthorTrialResult:
    """Replay author isotonic conduct using cohort-by-dose potential counts.

    Both outcome matrices have shape ``(cohorts, doses)``. At each cohort the
    column for the assigned dose is observed; other columns are unused
    potential cohort outcomes. The starting dose is always zero. The safety
    cap used to choose the next dose is the value from before the current
    cohort, matching the lag in the rendered ``isotonic()`` function.
    """
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    toxicities = _count_matrix(cohort_toxicities, "cohort_toxicities", size)
    responses = _count_matrix(cohort_responses, "cohort_responses", size)
    if toxicities.shape != responses.shape:
        raise ValueError("cohort outcome matrices must have matching shapes")
    cohorts, doses = toxicities.shape
    if cohorts * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1:
        raise ValueError("toxicity_limit must be finite and in (0,1)")
    if not 0 < cutoff < 1:
        raise ValueError("safety_cutoff must be finite and in (0,1)")

    n = np.zeros(doses, dtype=np.int64)
    y = np.zeros(doses, dtype=np.int64)
    r = np.zeros(doses, dtype=np.int64)
    current = 0
    cap = _mtadf_author_admissible_dose_count(n, y, toxicity_limit=phi, safety_cutoff=cutoff)
    assigned = np.empty(cohorts, dtype=np.int64)
    cap_before = np.empty(cohorts, dtype=np.int64)
    cap_after = np.empty(cohorts, dtype=np.int64)
    for cohort in range(cohorts):
        assigned[cohort] = current
        cap_before[cohort] = cap
        n[current] += size
        y[current] += toxicities[cohort, current]
        r[current] += responses[cohort, current]
        # The source computes movement using the previous cohort's cap, then
        # refreshes the cap after choosing the next dose.
        interim = _mtadf_author_decision_with_count(
            n,
            y,
            r,
            current_dose=current,
            final=False,
            toxicity_limit=phi,
            safety_cutoff=cutoff,
            admissibility_count=cap,
        )
        assert interim.dose is not None
        current = interim.dose
        cap = interim.admissible_dose_count
        cap_after[cohort] = cap

    final = mtadf_author_decision(
        n,
        y,
        r,
        final=True,
        toxicity_limit=phi,
        safety_cutoff=cutoff,
    )
    assert final.dose is not None
    return MTADFAuthorTrialResult(
        _freeze_int(n),
        _freeze_int(y),
        _freeze_int(r),
        _freeze_int(assigned),
        _freeze_int(cap_before),
        _freeze_int(cap_after),
        final.dose,
        final,
        "maximum_enrollment",
    )


def simulate_mtadf_author(
    true_toxicity: ArrayLike,
    true_efficacy: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 100,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    rng: int | np.integer | np.random.Generator | None = None,
    max_total_decisions: int = _MAX_TOTAL_DECISIONS,
    max_total_potential_cells: int = _MAX_TOTAL_POTENTIAL_CELLS,
) -> MTADFSimulation:
    """Run bounded serial author-convention trials and summarize outcomes.

    Each simulated trial draws independent cohort-binomial potential outcomes
    for every dose and endpoint, then observes only the assigned dose. This is
    a clear NumPy replay convention; it does not reproduce R's random stream.
    """
    tox = _probabilities(true_toxicity, "true_toxicity")
    eff = _probabilities(true_efficacy, "true_efficacy")
    if tox.shape != eff.shape:
        raise ValueError("true_toxicity and true_efficacy must have matching dose counts")
    cohort_count = _integer(cohorts, "cohorts", 1, _MAX_COHORTS)
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    if cohort_count * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    decision_limit = _integer(max_total_decisions, "max_total_decisions", 1, _MAX_TOTAL_DECISIONS)
    potential_limit = _integer(
        max_total_potential_cells,
        "max_total_potential_cells",
        1,
        _MAX_TOTAL_POTENTIAL_CELLS,
    )
    decision_work = trial_count * (cohort_count + 1)
    potential_work = trial_count * cohort_count * tox.size * 2
    if decision_work > decision_limit:
        raise ValueError("requested simulation exceeds max_total_decisions")
    if potential_work > potential_limit:
        raise ValueError("requested simulation exceeds max_total_potential_cells")
    # Validate conduct limits before touching RNG.
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    if isinstance(rng, np.random.Generator):
        generator = rng
    elif rng is None or (
        isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_))
    ):
        generator = np.random.default_rng(rng)
    else:
        raise ValueError("rng must be an integer seed, Generator, or None")

    n_by_trial = np.zeros((trial_count, tox.size), dtype=np.int64)
    y_by_trial = np.zeros_like(n_by_trial)
    r_by_trial = np.zeros_like(n_by_trial)
    selected: NDArray[np.int64] = np.empty(trial_count, dtype=np.int64)
    reasons: list[str] = []
    for trial in range(trial_count):
        potential_tox = np.column_stack(
            [generator.binomial(size, p, size=cohort_count) for p in tox]
        )
        potential_eff = np.column_stack(
            [generator.binomial(size, p, size=cohort_count) for p in eff]
        )
        result = replay_mtadf_author_trial(
            potential_tox,
            potential_eff,
            cohort_size=size,
            toxicity_limit=phi,
            safety_cutoff=cutoff,
        )
        n_by_trial[trial] = result.subjects
        y_by_trial[trial] = result.toxicities
        r_by_trial[trial] = result.responses
        selected[trial] = result.selected_dose
        reasons.append(result.stop_reason)

    frequencies: FloatArray = np.bincount(selected, minlength=tox.size).astype(float)
    selection = frequencies / trial_count
    selection_mcse = np.sqrt(selection * (1 - selection) / trial_count)
    mean_n = np.mean(n_by_trial, axis=0)
    mean_y = np.mean(y_by_trial, axis=0)
    mean_r = np.mean(r_by_trial, axis=0)
    zero = 0.0
    return MTADFSimulation(
        _freeze_int(n_by_trial),
        _freeze_int(y_by_trial),
        _freeze_int(r_by_trial),
        _freeze_int(selected),
        _freeze(selection),
        _freeze(selection_mcse),
        zero,
        zero,
        zero,
        zero,
        _freeze(mean_n),
        _freeze(mean_y),
        _freeze(mean_r),
        tuple(reasons),
    )
