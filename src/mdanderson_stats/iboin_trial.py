"""Deterministic patient-level replay for iBOIN accelerated titration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite
from .bayesian_monitoring import _integer
from .boin import BOINDecision, _owned
from .iboin import IBOINDesign

_MAX_PATIENTS = 100_000
_MAX_HISTORY_CELLS = 500_000


@dataclass(frozen=True)
class IBOINTrialDecision:
    """A BOIN decision and cumulative data immediately after its cohort."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    grade2_toxicities: NDArray[np.int64]
    current_dose: int
    decision: BOINDecision


@dataclass(frozen=True)
class IBOINTrialReplay:
    """Auditable patient outcomes, assignments and current enrollment state.

    ``next_dose`` is the dose for the next outcome row when outcomes are
    pending. It is ``None`` after a design stop or the explicit patient budget.
    ``cohort_remaining`` records the number still needed before the next BOIN
    decision; it remains nonzero when the budget stops an incomplete group.
    Titration grade-2 events are maximum-severity categories and are exclusive
    of DLTs. No final MTD selection is performed.
    """

    assigned_dose: NDArray[np.int64]
    dlt: NDArray[np.int64]
    grade2: NDArray[np.int64]
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    grade2_toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    next_dose: int | None
    phase: Literal["titration", "cohort", "stopped"]
    cohort_remaining: int
    stop_reason: str | None
    titration_end_reason: str | None
    decision_history: tuple[IBOINTrialDecision, ...]


def _conduct_settings(
    design: IBOINDesign,
    cohort_size: int,
    starting_dose: int,
    titration: bool,
    titration_cap: int | None,
    max_patients: int,
) -> tuple[int, int, int, bool, int, int]:
    if not isinstance(design, IBOINDesign):
        raise ValueError("design must be an IBOINDesign")
    cohort_count = count(cohort_size, "cohort_size")
    start_count = count(starting_dose, "starting_dose")
    n_doses = int(design.log_hypothesis_probability.shape[0])
    if cohort_count.ndim != 0 or cohort_count < 1:
        raise ValueError("cohort_size must be a positive integer")
    if start_count.ndim != 0 or not 1 <= start_count <= n_doses:
        raise ValueError("starting_dose must identify a design dose")
    max_count = _integer(max_patients, "max_patients")
    if not 1 <= max_count <= _MAX_PATIENTS:
        raise ValueError("max_patients must be in [1, 100000]")
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    if titration_cap is None:
        cap = n_doses
    else:
        cap_count = count(titration_cap, "titration_cap")
        if cap_count.ndim != 0 or not int(start_count) <= cap_count <= n_doses:
            raise ValueError("titration_cap must be between starting and highest dose")
        cap = int(cap_count)
    if not titration and titration_cap is not None:
        raise ValueError("titration_cap requires titration=True")
    return n_doses, int(cohort_count), int(start_count), bool(titration), cap, max_count


class _IBOINConductState:
    """Shared incremental conduct transition used by replay and simulation."""

    def __init__(
        self,
        design: IBOINDesign,
        n_doses: int,
        cohort_size: int,
        starting_dose: int,
        titration: bool,
        titration_cap: int,
        max_patients: int,
        *,
        record_history: bool = True,
    ) -> None:
        self.design = design
        self.n_doses = n_doses
        self.cohort_size = cohort_size
        self.max_patients = max_patients
        self.record_history = record_history
        self.assigned: list[int] = []
        self.dlt: list[int] = []
        self.grade2: list[int] = []
        self.patients = np.zeros(n_doses, dtype=np.int64)
        self.toxicities = np.zeros(n_doses, dtype=np.int64)
        self.moderate = np.zeros(n_doses, dtype=np.int64)
        self.excluded = np.zeros(n_doses, dtype=bool)
        self.decisions: list[IBOINTrialDecision] = []
        self.phase: Literal["titration", "cohort", "stopped"] = (
            "titration" if titration else "cohort"
        )
        self.current_dose = starting_dose
        self.next_dose: int | None = starting_dose
        self.cohort_remaining = 1 if titration else cohort_size
        self.stop_reason: str | None = None
        self.titration_end_reason: str | None = None
        self.titration_cap = titration_cap

    def observe(self, dlt: int, grade2: int) -> int:
        if self.next_dose is None:
            raise ValueError("outcomes were supplied after the design stopped")
        current_dose = self.next_dose
        j = current_dose - 1
        self.assigned.append(current_dose)
        self.dlt.append(dlt)
        self.grade2.append(grade2)
        self.patients[j] += 1
        self.toxicities[j] += dlt
        self.moderate[j] += grade2

        if self.phase == "titration":
            if dlt:
                self.titration_end_reason = "DLT"
                self.phase = "cohort"
                self.cohort_remaining = self.cohort_size - 1
            elif int(np.sum(self.moderate)) >= 2:
                self.titration_end_reason = "grade2"
                self.phase = "cohort"
                self.cohort_remaining = self.cohort_size - 1
            elif current_dose == self.n_doses:
                self.titration_end_reason = "highest_dose"
                self.phase = "cohort"
                self.cohort_remaining = self.cohort_size - 1
            elif current_dose == self.titration_cap:
                self.titration_end_reason = "dose_cap"
                self.current_dose = current_dose + 1
                self.phase = "cohort"
                self.cohort_remaining = self.cohort_size
            else:
                self.current_dose = current_dose + 1
                self.cohort_remaining = 1
        else:
            self.cohort_remaining -= 1

        if self.phase == "cohort" and self.cohort_remaining == 0:
            decision = self.design.next_dose(
                self.patients,
                self.toxicities,
                self.current_dose,
                eliminated=self.excluded,
            )
            if self.record_history:
                self.decisions.append(
                    IBOINTrialDecision(
                        _owned(self.patients),
                        _owned(self.toxicities),
                        _owned(self.moderate),
                        self.current_dose,
                        decision,
                    )
                )
            self.excluded = np.asarray(decision.eliminated, dtype=bool).copy()
            if decision.next_dose is None:
                self.stop_reason = decision.action
                self.phase = "stopped"
                self.next_dose = None
                self.cohort_remaining = 0
            else:
                self.current_dose = decision.next_dose
                self.next_dose = self.current_dose
                self.cohort_remaining = self.cohort_size

        if self.stop_reason is None and len(self.assigned) == self.max_patients:
            self.stop_reason = "stop_max_patients"
            self.phase = "stopped"
            self.next_dose = None
        elif self.stop_reason is None:
            self.next_dose = self.current_dose
        return current_dose

    def result(self) -> IBOINTrialReplay:
        return IBOINTrialReplay(
            _owned(np.asarray(self.assigned, dtype=np.int64)),
            _owned(np.asarray(self.dlt, dtype=np.int64)),
            _owned(np.asarray(self.grade2, dtype=np.int64)),
            _owned(self.patients),
            _owned(self.toxicities),
            _owned(self.moderate),
            _owned(self.excluded),
            self.next_dose,
            self.phase,
            self.cohort_remaining,
            self.stop_reason,
            self.titration_end_reason,
            tuple(self.decisions),
        )


def replay_iboin_trial(
    design: IBOINDesign,
    dlt: ArrayLike,
    grade2: ArrayLike,
    *,
    cohort_size: int,
    starting_dose: int = 1,
    titration: bool = True,
    titration_cap: int | None = None,
    max_patients: int,
) -> IBOINTrialReplay:
    """Assign doses and replay complete patient outcomes under iBOIN conduct.

    Outcome rows are consumed in enrollment order, at the dose currently
    assigned by the protocol. During titration, patients escalate one dose at a
    time until the first DLT, second grade-2 event, highest dose, or a lower
    titration cap. A toxicity/highest-dose trigger tops up the current dose to
    the cohort size; a lower cap without a trigger starts a full cohort one
    dose higher. Thereafter, ``design.next_dose`` is called only at complete
    cohort boundaries.

    If the supplied outcomes end before ``max_patients``, the returned replay
    is pending and exposes the next assigned dose and cohort remainder. The
    required ``max_patients`` is a hard terminal enrollment budget; reaching it
    suppresses further assignment even when the last cohort is incomplete.
    Design safety or precision stops take precedence when they occur at that
    exact boundary. Outcomes supplied after a design stop are rejected.

    This is a conduct/replay API, not a random simulator or final MTD selector.
    Grade-2 values encode mutually exclusive maximum severity: a patient cannot
    be both DLT and grade-2 in the same row.
    """
    n_doses, cohort_count, start_count, use_titration, cap, max_count = _conduct_settings(
        design, cohort_size, starting_dose, titration, titration_cap, max_patients
    )

    if np.iscomplexobj(dlt) or np.iscomplexobj(grade2):
        raise ValueError("patient outcomes must be real")
    raw_dlt, raw_grade2 = np.asarray(dlt), np.asarray(grade2)
    if (
        raw_dlt.ndim != 1
        or raw_grade2.shape != raw_dlt.shape
        or raw_dlt.size > int(max_count)
        or raw_dlt.size > _MAX_PATIENTS
        or raw_dlt.dtype.kind not in "biuf"
        or raw_grade2.dtype.kind not in "biuf"
    ):
        raise ValueError("dlt and grade2 must be aligned outcome vectors within the patient budget")
    n_observed = int(raw_dlt.size)
    if n_observed * n_doses > _MAX_HISTORY_CELLS:
        raise ValueError("decision history exceeds the bounded outcome-by-dose limit")
    dlt_values = finite(raw_dlt, "dlt")
    grade2_values = finite(raw_grade2, "grade2")
    if (
        np.any((dlt_values != 0) & (dlt_values != 1))
        or np.any((grade2_values != 0) & (grade2_values != 1))
        or np.any((dlt_values == 1) & (grade2_values == 1))
    ):
        raise ValueError("dlt and grade2 must be exclusive binary maximum-severity outcomes")
    dlt_values = dlt_values.astype(np.int64)
    grade2_values = grade2_values.astype(np.int64)

    state = _IBOINConductState(
        design, n_doses, cohort_count, start_count, use_titration, cap, max_count
    )
    for index, (dlt_value, grade2_value) in enumerate(zip(dlt_values, grade2_values, strict=True)):
        if state.stop_reason is not None:
            raise ValueError("outcomes were supplied after the design stopped")
        state.observe(int(dlt_value), int(grade2_value))
        if state.stop_reason is not None and index + 1 < n_observed:
            raise ValueError("outcomes were supplied after the design stopped")
    return state.result()
