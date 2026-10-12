"""Current EffTox cohort and pending controller, with explicit inference service.

The supplied service must deterministically analyze a complete count matrix.
This separates source-verified controller behavior from posterior approximation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import comb, prod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .efftox_desktop import EffToxDesktopStudy
from .efftox_desktop_decision import EffToxDesktopDecision
from .efftox_desktop_inference import EffToxDesktopInference
from .efftox_model import _integer

Status = Literal["go", "wait", "stop_early", "complete", "work_limit"]
Analysis = Callable[[NDArray[np.int64]], EffToxDesktopDecision]


def _cohort_prefix(cohorts: Sequence[int], size: int) -> tuple[int, bool, int]:
    if not cohorts:
        return 0, True, 1
    prefix = len(cohorts) - 1
    while prefix > 0 and cohorts[prefix - 1] == cohorts[-1]:
        prefix -= 1
    new = len(cohorts) - prefix == size
    return (len(cohorts) if new else prefix), new, cohorts[-1] + int(new)


def _pending_consistent(
    baseline: tuple[int, bool], completions: Iterable[tuple[int, bool]]
) -> bool:
    # Original loop compares every completion against the observed-only baseline.
    return all(completion == baseline for completion in completions)


def _status(patients: int, maximum: int, consistent: bool, stopped: bool) -> Status:
    if not consistent:
        return "wait"
    if stopped:
        return "stop_early"
    return "complete" if patients >= maximum else "go"


def _compositions(n: int) -> Iterator[tuple[int, int, int, int]]:
    for a in range(n + 1):
        for b in range(n - a + 1):
            for c in range(n - a - b + 1):
                yield a, b, c, n - a - b - c


def _completions(
    counts: NDArray[np.int64], pending: NDArray[np.int64], dose: int = 0
) -> Iterator[NDArray[np.int64]]:
    # Aggregate exchangeable patients by administered dose. This enumerates the
    # same possible sufficient statistics, not the original enumerator's order.
    if dose == len(pending):
        yield counts.copy()
        return
    for cells in _compositions(int(pending[dose])):
        updated = counts.copy()
        updated[dose] += np.asarray(cells).reshape(2, 2)
        yield from _completions(updated, pending, dose + 1)


@dataclass(frozen=True)
class EffToxDesktopRecommendation:
    """Controller output; dose indices are zero based, never enrollment orders.

    ``capacity_guard`` identifies the deliberate safe deviation from the desktop
    partial-final-cohort bug. ``analyses`` includes the observed-only baseline.
    No dose is released on a wait or work-limit result. A complete result can
    contain the final selected dose. No clinical validity is asserted.
    """

    status: Status
    dose_index: int | None
    baseline: EffToxDesktopDecision | None
    analyzed_patients: int
    pending_patients: int
    completion_count: int
    analyses: int
    new_cohort: bool
    next_cohort: int
    capacity_guard: bool


def recommend_efftox_desktop(
    study: EffToxDesktopStudy,
    analyze: Analysis,
    *,
    max_analyses: int = 1000,
    capacity_profile: Literal["safe", "desktop_legacy"] = "safe",
) -> EffToxDesktopRecommendation:
    """Apply current GUI cohort/pending rules using an explicit inference service.

    All four outcomes of each pending patient are checked through distinct count
    matrices. The full bound is checked before inference; exceeding it produces
    ``work_limit``, never a recommendation from a truncated search. At capacity,
    the safe profile analyzes the complete history, including a partial final
    cohort. ``desktop_legacy`` explicitly retains the original prefix/capacity
    defect and can return GO at capacity; it must not authorize enrollment.
    """
    if not isinstance(study, EffToxDesktopStudy) or not callable(analyze):
        raise TypeError("expected an EffToxDesktopStudy and callable analysis service")
    if float(ET.fromstring(study.xml).findtext("TrialIsShutDown", "nan")) != 0:
        raise ValueError("the saved trial is shut down")
    limit = _integer(max_analyses, "max_analyses", 1, 100_000)
    if capacity_profile not in {"safe", "desktop_legacy"}:
        raise ValueError("unknown capacity_profile")
    prefix, new, next_cohort = _cohort_prefix([p.cohort for p in study.patients], study.cohort_size)
    guard = (
        capacity_profile == "safe"
        and len(study.patients) >= study.max_patients
        and prefix != len(study.patients)
    )
    if guard:
        prefix = len(study.patients)
    counts = np.zeros((len(study.doses), 2, 2), dtype=np.int64)
    pending = np.zeros(len(study.doses), dtype=np.int64)
    for patient in study.patients[:prefix]:
        outcome = patient.binary_outcome
        if outcome is None:
            pending[patient.administered_dose_index] += 1
        else:
            counts[patient.administered_dose_index, outcome[0], outcome[1]] += 1
    quantity = int(pending.sum())
    completions = prod(comb(int(n) + 3, 3) for n in pending) if quantity else 0
    if completions + 1 > limit:
        return EffToxDesktopRecommendation(
            "work_limit", None, None, prefix, quantity, completions, 0, new, next_cohort, guard
        )
    if isinstance(analyze, EffToxDesktopInference):
        analyze.check_study(study)
        analyze.require_analyses(completions + 1)
    analyses = 0

    def evaluate(matrix: NDArray[np.int64]) -> EffToxDesktopDecision:
        nonlocal analyses
        matrix.flags.writeable = False
        result = analyze(matrix)
        if not isinstance(result, EffToxDesktopDecision):
            raise TypeError("analysis service must return EffToxDesktopDecision")
        if result.dose_index is not None:
            _integer(result.dose_index, "selected dose", 0, len(study.doses) - 1)
        analyses += 1
        return result

    baseline = evaluate(counts.copy())

    def selections() -> Iterator[tuple[int, bool]]:
        for matrix in _completions(counts, pending):
            decision = evaluate(matrix)
            yield (
                -1 if decision.dose_index is None else decision.dose_index,
                decision.dose_index is None,
            )

    key = (-1 if baseline.dose_index is None else baseline.dose_index, baseline.dose_index is None)
    consistent = not quantity or _pending_consistent(key, selections())
    status = _status(prefix, study.max_patients, consistent, baseline.dose_index is None)
    return EffToxDesktopRecommendation(
        status,
        baseline.dose_index if consistent else None,
        baseline,
        prefix,
        quantity,
        completions,
        analyses,
        new,
        next_cohort,
        guard,
    )
