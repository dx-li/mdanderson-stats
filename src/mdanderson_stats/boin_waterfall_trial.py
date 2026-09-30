"""Deterministic replay of complete-outcome BOIN waterfall subtrials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betaincc

from ._validation import FloatArray
from .boin import BOINBoundaryTable, _owned
from .boin_combination import BOINCombDesign
from .keyboard_combination import _biviso

DoseCombination = tuple[int, int]
_MAX_DIMENSION = 20
_MAX_PATIENTS = 1000
_MAX_CELLS = 400


def _frozen_counts(value: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _matrix(value: ArrayLike, name: str, *, probability: bool = False) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.size > _MAX_CELLS or not 2 <= raw.shape[0] <= raw.shape[1]:
        raise ValueError(f"{name} must be a 2..20 row-by-column matrix with rows <= columns")
    if raw.shape[1] > _MAX_DIMENSION or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real matrix with at most 20 columns")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    if probability and np.any((result < 0) | (result > 1)):
        raise ValueError("true_toxicity must contain probabilities in [0,1]")
    return result


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return result


def _budgets(value: ArrayLike, rows: int, cohort_size: int) -> tuple[tuple[int, ...], int]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != rows or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("cohort_budgets must be an integer vector with one entry per dose row")
    if np.any(raw < 1):
        raise ValueError("each subtrial cohort budget must be positive")
    if any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError("cohort_budgets must contain integers")
    result = tuple(int(item) for item in raw)
    maximum = sum(result) * cohort_size
    if maximum > _MAX_PATIENTS:
        raise ValueError("the waterfall trial's planned enrollment must not exceed 1000 patients")
    return result, maximum


def _initial_space(rows: int, columns: int) -> tuple[DoseCombination, ...]:
    return tuple([(i, 1) for i in range(1, rows + 1)] + [(rows, j) for j in range(2, columns + 1)])


def _row_space(row: int, columns: int) -> tuple[DoseCombination, ...]:
    return tuple((row, j) for j in range(2, columns + 1))


@dataclass(frozen=True)
class WaterfallCohort:
    """One assigned cohort in its executed subtrial."""

    subtrial: int
    cohort: int
    dose: DoseCombination
    patients: int
    toxicities: int
    phase: str = "cohort"


@dataclass(frozen=True)
class WaterfallSubtrial:
    """Executed slice and cumulative snapshots.

    The snapshot's exclusion matrix also reflects final Beta(1,1) selector
    safety checks. This makes final-only extra-safe rejection visible, whereas
    the original nested routine returned its conduct-only mask.
    """

    index: int
    dose_space: tuple[DoseCombination, ...]
    start_position: int
    cohort_budget: int
    cohorts: tuple[WaterfallCohort, ...]
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    candidate: DoseCombination | None
    escalation_allowed: bool | None
    stop_reason: str


@dataclass(frozen=True)
class BOINWaterfallTrial:
    """Actual trial counts, full subtrial trace and final row-wise contour.

    ``row_candidates`` stores the candidates before source continuity is
    applied. ``selected_contour`` follows that continuity when it remains on
    an observed, non-eliminated cell; otherwise that row is left unselected and
    ``continuity_blocked`` flags that row. ``source_contour`` records the
    destination before this admissibility check.
    """

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    subtrials: tuple[WaterfallSubtrial, ...]
    cohort_history: tuple[WaterfallCohort, ...]
    row_candidates: tuple[DoseCombination | None, ...]
    source_contour: tuple[DoseCombination | None, ...]
    selected_contour: tuple[DoseCombination | None, ...]
    continuity_blocked: NDArray[np.bool_]
    isotonic_estimate: FloatArray
    total_patients: int
    total_toxicities: int
    stop_reason: str
    titration_patients: NDArray[np.int64]
    titration_endpoint: DoseCombination | None
    titration_end_reason: str


def _slice_candidate(
    design: BOINCombDesign,
    patients: NDArray[np.int64],
    toxicities: NDArray[np.int64],
    excluded: NDArray[np.bool_],
    *,
    extra_safe: bool,
    offset: float,
    bound_mtd: bool,
    boundary_table: BOINBoundaryTable,
) -> tuple[int | None, bool | None, NDArray[np.bool_], str]:
    """Reproduce the waterfall subtrial's weak-prior one-dimensional choice."""
    n = patients
    y = toxicities
    elimination = excluded.copy()
    for index in range(n.size):
        if (
            n[index] >= 3
            and betaincc(y[index] + 1, n[index] - y[index] + 1, design.target)
            > design.elimination_probability
        ):
            elimination[index:] = True
            break
    if extra_safe and n[0] >= 3:
        if (
            betaincc(
                y[0] + 1,
                n[0] - y[0] + 1,
                design.target,
            )
            > design.elimination_probability - offset
        ):
            elimination[:] = True
    if elimination[0] or not np.any((n > 0) & ~elimination):
        return None, None, elimination, "stop_safety" if elimination[0] else "stop_no_data"

    active = np.flatnonzero((n > 0) & ~elimination)
    na, ya = n[active].astype(float), y[active].astype(float)
    estimate = (ya + 0.05) / (na + 0.1)
    variance = (ya + 0.05) * (na - ya + 0.05) / ((na + 0.1) ** 2 * (na + 1.1))
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ArithmeticError("subtrial posterior isotonic weights are not representable")
    fitted = isotonic_regression(estimate, weights=1.0 / variance).x
    fitted += np.arange(1, fitted.size + 1) * 1e-10
    candidates = np.arange(active.size)
    if bound_mtd:
        candidates = candidates[fitted < design.deescalation_boundary]
    if candidates.size == 0:
        return None, None, elimination, "no_candidate_under_bound"
    winner = int(candidates[np.argmin(np.abs(fitted[candidates] - design.target))])
    position = int(active[winner])
    escalation = bool(y[position] <= boundary_table.escalate_max[int(n[position]) - 1])
    return position, escalation, elimination, "selected"


def _final_contour(
    design: BOINCombDesign,
    patients: NDArray[np.int64],
    toxicities: NDArray[np.int64],
    eliminated: NDArray[np.bool_],
) -> tuple[
    tuple[DoseCombination | None, ...],
    tuple[DoseCombination | None, ...],
    tuple[DoseCombination | None, ...],
    NDArray[np.bool_],
    FloatArray,
]:
    n = patients.astype(float)
    y = toxicities.astype(float)
    values = (y + 0.05) / (n + 0.1)
    values[eliminated] = 1.1
    fitted = _biviso(values, n + 0.1)
    tie_adjusted = fitted + 1e-5 * (np.indices(n.shape).sum(axis=0) + 2)
    candidates: list[DoseCombination | None] = []
    for i in range(n.shape[0]):
        active = np.flatnonzero((n[i] > 0) & ~eliminated[i])
        if eliminated[i, 0] or not np.any((n[~eliminated]) > 0) or active.size == 0:
            candidates.append(None)
            continue
        j = int(active[np.argmin(np.abs(tie_adjusted[i, active] - design.target))])
        candidates.append((i + 1, j + 1))

    source = list(candidates)
    for i in range(n.shape[0] - 2, -1, -1):
        lower = source[i + 1]
        current = candidates[i]
        if lower is None or current is None or current[1] > lower[1]:
            continue
        source[i] = (i + 1, lower[1])
    selected = list(source)
    blocked = np.zeros(n.shape[0], dtype=bool)
    for i, pair in enumerate(source):
        if pair is None:
            continue
        row, column = pair[0] - 1, pair[1] - 1
        if n[row, column] > 0 and not eliminated[row, column]:
            continue
        selected[i] = None
        blocked[i] = True
    return tuple(candidates), tuple(source), tuple(selected), _freeze_bool(blocked), _owned(fitted)


def _freeze_bool(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    result = np.array(values, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def run_boin_waterfall_trial(
    design: BOINCombDesign,
    true_toxicity: ArrayLike,
    cohort_budgets: ArrayLike,
    outcome_uniforms: ArrayLike,
    *,
    cohort_size: int = 3,
    start_dose: int = 1,
    early_stop_patients: int = 12,
    bound_mtd: bool = False,
    titration: bool = False,
) -> BOINWaterfallTrial:
    """Replay one waterfall trial from an assigned-patient uniform tape.

    The tape is consumed sequentially in assignment order, cohort by cohort;
    with titration enabled, the first ``len(staircase)`` entries are reserved
    for the full source-style single-patient staircase draw, and later entries
    are used for cohort enrollment. Dose labels and search spaces are one-based.
    The native waterfall wrapper starts titration at staircase position one
    even when ``start_dose`` is supplied; this replay follows that convention.
    Cohort size one disables titration as in the source. Python applies the
    extra-safe interim rule whenever the first active dose has at least three
    patients; the R wrapper nests that check under a nonmissing standard
    elimination boundary. The final selector also uses its distinct uniform
    prior for its extra-safe check.
    """
    if not isinstance(design, BOINCombDesign):
        raise ValueError("design must be a BOINCombDesign")
    probability = _matrix(true_toxicity, "true_toxicity", probability=True)
    rows, columns = probability.shape
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    budgets, maximum = _budgets(cohort_budgets, rows, size)
    early = _integer(early_stop_patients, "early_stop_patients", 1, _MAX_PATIENTS)
    if not isinstance(bound_mtd, (bool, np.bool_)):
        raise ValueError("bound_mtd must be boolean")
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    space = _initial_space(rows, columns)
    requested_start = _integer(start_dose, "start_dose", 1, len(space)) - 1
    use_titration = bool(titration) and size > 1
    initial_position = 0 if use_titration else requested_start
    prelude_size = len(space) if use_titration else 0
    required_tape = maximum + (prelude_size - 1 if use_titration else 0)
    raw_tape = np.asarray(outcome_uniforms)
    if raw_tape.ndim != 1 or raw_tape.size != required_tape or raw_tape.dtype.kind not in "iuf":
        raise ValueError(f"outcome_uniforms must be a real vector of length {required_tape}")
    tape = np.asarray(raw_tape, dtype=float)
    if not np.all(np.isfinite(tape)) or np.any((tape < 0) | (tape >= 1)):
        raise ValueError("outcome_uniforms must be finite values in [0,1)")
    maximum_actual = maximum + (prelude_size - 1 if use_titration else 0)
    if maximum_actual > _MAX_PATIENTS:
        raise ValueError("planned enrollment plus titration overhead must not exceed 1000 patients")
    boundary_table = design.boundary_table(maximum_actual)

    patients = np.zeros((rows, columns), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    eliminated = np.zeros((rows, columns), dtype=bool)
    all_cohorts: list[WaterfallCohort] = []
    subtrials: list[WaterfallSubtrial] = []
    titration_patients = np.zeros_like(patients)
    titration_endpoint: DoseCombination | None = None
    titration_end_reason = "cohort_size_one" if bool(titration) and size == 1 else "not_requested"
    cursor = 0
    first_local_n: NDArray[np.int64] | None = None
    first_local_y: NDArray[np.int64] | None = None
    if use_titration:
        titration_events = tape[:prelude_size] < np.asarray(
            [probability[row - 1, column - 1] for row, column in space]
        )
        first_event = np.flatnonzero(titration_events)
        endpoint_position = int(first_event[0]) if first_event.size else len(space) - 1
        titration_end_reason = "first_dlt" if first_event.size else "upper_dose_no_dlt"
        titration_endpoint = space[endpoint_position]
        first_local_n = np.zeros(len(space), dtype=np.int64)
        first_local_y = np.zeros(len(space), dtype=np.int64)
        first_local_n[: endpoint_position + 1] = 1
        if first_event.size:
            first_local_y[endpoint_position] = 1
        for position, dose in enumerate(space[: endpoint_position + 1]):
            event = int(bool(titration_events[position]))
            row, column = dose
            patients[row - 1, column - 1] += 1
            toxicities[row - 1, column - 1] += event
            titration_patients[row - 1, column - 1] += 1
            all_cohorts.append(WaterfallCohort(1, position + 1, dose, 1, event, "titration"))
        cursor = prelude_size
        initial_position = endpoint_position
    stop_reason = "completed_search"
    fallback_used = False
    # Descriptor fields: search space, start index, budget index, special same-row
    # branch, and the candidate used if that branch cannot select.
    queue: list[tuple[tuple[DoseCombination, ...], int, int, bool, DoseCombination | None]] = [
        (space, initial_position, 0, False, None)
    ]
    while queue:
        next_space, next_position, budget_index, special, fallback_candidate = queue.pop(0)
        if budget_index >= len(budgets):
            stop_reason = "subtrial_budget_exhausted"
            break
        subtrial_index = len(subtrials) + 1
        budget = budgets[budget_index]
        if use_titration and subtrial_index == 1:
            if first_local_n is None or first_local_y is None:
                raise ArithmeticError("missing initial titration counts")
            local_n = first_local_n.copy()
            local_y = first_local_y.copy()
        else:
            local_n = np.zeros(len(next_space), dtype=np.int64)
            local_y = np.zeros_like(local_n)
        local_eliminated = np.zeros(len(next_space), dtype=bool)
        position = next_position
        trace = [item for item in all_cohorts if item.subtrial == subtrial_index]
        local_stop = "cohort_budget"
        for cohort in range(1, budget + 1):
            row, column = next_space[position]
            p = probability[row - 1, column - 1]
            top_up = use_titration and subtrial_index == 1 and cohort == 1
            cohort_patients = size - 1 if top_up else size
            sl = tape[cursor : cursor + cohort_patients]
            if sl.size != cohort_patients:
                raise ArithmeticError("outcome tape exhausted before the planned cohort")
            events = int(np.count_nonzero(sl < p))
            cursor += cohort_patients
            local_n[position] += cohort_patients
            local_y[position] += events
            patients[row - 1, column - 1] += cohort_patients
            toxicities[row - 1, column - 1] += events
            phase = "titration_topup" if top_up else "cohort"
            record = WaterfallCohort(
                subtrial_index, cohort, (row, column), cohort_patients, events, phase
            )
            trace.append(record)
            all_cohorts.append(record)

            n_here = int(local_n[position])
            if local_y[position] >= boundary_table.eliminate_min[n_here - 1]:
                local_eliminated[position:] = True
                for r, c in next_space[position:]:
                    eliminated[r - 1, c - 1] = True
                if position == 0:
                    local_stop = "stop_safety"
                    break
            if design.extra_safe and position == 0 and n_here >= 3:
                extra_tail = betaincc(
                    local_y[0] + 0.05,
                    local_n[0] - local_y[0] + 0.1,
                    design.target,
                )
                if extra_tail > design.elimination_probability - design.safety_offset:
                    local_stop = "stop_extra_safe"
                    break

            if (
                local_y[position] <= boundary_table.escalate_max[n_here - 1]
                and position != len(next_space) - 1
            ):
                if not local_eliminated[position + 1]:
                    position += 1
            elif local_y[position] >= boundary_table.deescalate_min[n_here - 1] and position != 0:
                position -= 1
            if local_n[position] >= early:
                local_stop = "precision_at_destination"
                break
            if int(local_n.sum()) >= budget * size:
                local_stop = "cohort_budget"
                break

        if local_stop in ("stop_safety", "stop_extra_safe"):
            local_eliminated[:] = True
        candidate_pos: int | None = None
        escalation: bool | None = None
        if local_stop not in ("stop_safety", "stop_extra_safe"):
            candidate_pos, escalation, local_eliminated, selection_status = _slice_candidate(
                design,
                local_n,
                local_y,
                local_eliminated,
                extra_safe=design.extra_safe,
                offset=design.safety_offset,
                bound_mtd=bound_mtd,
                boundary_table=boundary_table,
            )
            if candidate_pos is None:
                local_stop = selection_status
        for index, (r, c) in enumerate(next_space):
            eliminated[r - 1, c - 1] |= local_eliminated[index]
        candidate = None if candidate_pos is None else next_space[candidate_pos]
        subtrials.append(
            WaterfallSubtrial(
                subtrial_index,
                next_space,
                1 if use_titration and subtrial_index == 1 else next_position + 1,
                budget,
                tuple(trace),
                _frozen_counts(patients),
                _frozen_counts(toxicities),
                _freeze_bool(eliminated),
                candidate,
                escalation,
                local_stop,
            )
        )
        if candidate is None and not special:
            stop_reason = local_stop
            break
        if candidate is None:
            fallback_used = True
            candidate = fallback_candidate
        if candidate is None:
            stop_reason = local_stop
            break
        row, column = candidate

        # First-subtrial special branch: search columns 2..K in the same row
        # when the selected first-column dose is below the last row and permits
        # escalation. If this extra search cannot select, preserve the initial
        # candidate for navigation, as the R wrapper does, but keep its counts.
        if not special and subtrial_index == 1 and column == 1 and row < rows:
            for lower_row in range(row + 1, rows + 1):
                eliminated[lower_row - 1, :] = True
            if escalation:
                same_row = _row_space(row, columns)
                if budget_index + 1 < len(budgets):
                    queue.insert(0, (same_row, 0, budget_index + 1, True, candidate))
                    continue
                stop_reason = "subtrial_budget_exhausted"
                break

        if row == 1:
            stop_reason = "first_row_reached"
            break
        if column < columns:
            eliminated[row - 1, column:] = True
        if use_titration and int(patients.sum()) >= maximum:
            stop_reason = "planned_enrollment_reached"
            break
        next_space = _row_space(row - 1, columns)
        # A row search begins at column 2. After selecting column j, begin
        # one column higher, at local index j-1, clipped to the final column.
        next_position = min(column - 1, len(next_space) - 1)
        if budget_index + 1 < len(budgets):
            queue.insert(0, (next_space, next_position, budget_index + 1, False, None))
        else:
            stop_reason = "subtrial_budget_exhausted"
            break

    if fallback_used:
        stop_reason = f"{stop_reason}_after_special_fallback"

    row_candidates, source_contour, contour, blocked, fitted = _final_contour(
        design, patients, toxicities, eliminated
    )
    return BOINWaterfallTrial(
        _frozen_counts(patients),
        _frozen_counts(toxicities),
        _freeze_bool(eliminated),
        tuple(subtrials),
        tuple(all_cohorts),
        row_candidates,
        source_contour,
        contour,
        blocked,
        fitted,
        int(patients.sum()),
        int(toxicities.sum()),
        stop_reason,
        _frozen_counts(titration_patients),
        titration_endpoint,
        titration_end_reason,
    )
