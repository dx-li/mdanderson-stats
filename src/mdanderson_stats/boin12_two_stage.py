"""Two-stage toxicity-only to BOIN12 utility dose finding.

The cached BOIN12 application specifies a caller-selected threshold S and
switches after a cohort when any dose has at least S treated patients.  The
triggering cohort is assigned under Stage 1; Stage 2 governs the next
assignment.  Stage 1 uses BOIN toxicity movement boundaries and the BOIN12
posterior safety cutoff, without a minimum-sample-size safety guard.  These
calendar and ordering details are explicit Python conduct policies, not claims
of exact parity with the native application.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincc

from ._validation import FloatArray, count, scalar
from .boin import _owned
from .boin12 import (
    BOIN12Decision,
    BOIN12Design,
    BOIN12Posterior,
    _additive_utilities,
    _counts,
    _joint_counts,
    admissibility,
)
from .boin12_simulation import _joint_grid as _validated_joint_grid
from .boin12_simulation import _start


@dataclass(frozen=True)
class BOIN12TwoStageDecision:
    """Decision state; Stage 1 has no utility posterior by design."""

    stage: int
    action: str
    next_dose: int | None
    admissible: NDArray[np.bool_]
    eliminated: NDArray[np.bool_]
    posterior: BOIN12Posterior | None


@dataclass(frozen=True)
class BOIN12TwoStageSimulation:
    """Trial-level counts and stage-transition diagnostics."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    efficacies: NDArray[np.int64]
    efficacy_without_toxicity: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_obd: NDArray[np.int64]
    selected_mtd: NDArray[np.int64]
    obd_probability: FloatArray
    obd_mcse: FloatArray
    mtd_probability: FloatArray
    mtd_mcse: FloatArray
    stop_reason: tuple[str, ...]
    transition_cohort: NDArray[np.int64]
    stage1_cohorts: NDArray[np.int64]
    stage2_cohorts: NDArray[np.int64]


def _threshold(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 6 <= int(value) <= 12
    ):
        raise ValueError("stage1_threshold must be an integer in [6,12]")
    return int(value)


def _validate_design(design: BOIN12Design) -> None:
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    for name in ("toxicity_cutoff", "efficacy_cutoff"):
        cutoff = scalar(getattr(design, name), name)
        if not 0 < cutoff < 1:
            raise ValueError(f"{name} must lie in (0,1)")


def _stage1_decision(
    design: BOIN12Design,
    patients: NDArray[np.float64],
    toxicities: NDArray[np.float64],
    current_dose: int,
    eliminated: NDArray[np.bool_],
) -> BOIN12TwoStageDecision:
    n, t = patients, toxicities
    phi = design.toxicity_limit
    overdose = betaincc(t + 1.0, n - t + 1.0, phi)
    # Match BOIN12's strict admissibility rule. The native help does not state
    # a minimum n; exclusions persist over time but are not propagated by dose.
    unsafe = overdose >= design.toxicity_cutoff
    updated = eliminated | unsafe
    all_allowed = ~updated
    index = current_dose - 1
    local = np.zeros(n.shape, dtype=bool)
    local[max(0, index - 1) : min(n.size, index + 2)] = True
    local &= all_allowed
    if not np.any(all_allowed):
        return BOIN12TwoStageDecision(1, "stop_safety", None, _owned(local), _owned(updated), None)
    if index == 0 and updated[index]:
        return BOIN12TwoStageDecision(1, "stop_safety", None, _owned(local), _owned(updated), None)
    if design.early_stop_patients is not None and n[index] >= design.early_stop_patients:
        return BOIN12TwoStageDecision(
            1, "stop_precision", None, _owned(local), _owned(updated), None
        )
    if updated[index]:
        next_index = index - 1
        if not all_allowed[next_index]:
            return BOIN12TwoStageDecision(
                1,
                "stop_no_admissible_neighbor",
                None,
                _owned(local),
                _owned(updated),
                None,
            )
        return BOIN12TwoStageDecision(
            1, "deescalate", next_index + 1, _owned(local), _owned(updated), None
        )

    rate = t[index] / n[index]
    move = (
        1
        if rate <= design._boin.escalation_boundary
        else -1
        if rate >= design._boin.deescalation_boundary
        else 0
    )
    next_index = max(0, min(index + move, n.size - 1))
    if updated[next_index]:
        if move < 0:
            return BOIN12TwoStageDecision(
                1,
                "stop_no_admissible_neighbor",
                None,
                _owned(local),
                _owned(updated),
                None,
            )
        next_index = index
    action = "escalate" if next_index > index else "deescalate" if next_index < index else "stay"
    return BOIN12TwoStageDecision(1, action, next_index + 1, _owned(local), _owned(updated), None)


def boin12_two_stage_next_dose(
    design: BOIN12Design,
    patients: ArrayLike,
    toxicities: ArrayLike,
    efficacies: ArrayLike,
    current_dose: int,
    *,
    stage1_threshold: int,
    efficacy_without_toxicity: ArrayLike | None = None,
    eliminated: ArrayLike | None = None,
) -> BOIN12TwoStageDecision:
    """Choose the next dose after a fully observed cohort.

    The current counts determine the stage for the next cohort. Thus, a cohort
    that first reaches ``stage1_threshold`` is the final Stage 1 cohort. In
    Stage 1, only BOIN toxicity boundaries and the BOIN12 safety posterior are
    used; Stage 2 delegates to :meth:`BOIN12Design.next_dose`. Safety is
    evaluated before the optional patient-count precision stop.
    """
    _validate_design(design)
    threshold = _threshold(stage1_threshold)
    n, t, e = _counts(patients, toxicities, efficacies)
    if efficacy_without_toxicity is not None or not _additive_utilities(
        np.asarray(design.utilities)
    ):
        _joint_counts(
            n,
            t,
            e,
            efficacy_without_toxicity,
            np.asarray(design.utilities),
        )
    if isinstance(current_dose, (bool, np.bool_)) or int(current_dose) != current_dose:
        raise ValueError("current_dose must be an integer dose index")
    current = int(current_dose)
    if not 1 <= current <= n.size or n[current - 1] == 0:
        raise ValueError("current_dose must identify a treated dose")
    excluded = np.zeros(n.shape, dtype=bool)
    if eliminated is not None:
        supplied = np.asarray(eliminated)
        if supplied.shape != n.shape or supplied.dtype != np.bool_:
            raise ValueError("eliminated must be a matching boolean vector")
        excluded = supplied.copy()
    if np.any(n >= threshold):
        posterior = design.posterior(n, t, e, efficacy_without_toxicity=efficacy_without_toxicity)
        excluded |= ~admissibility(
            posterior,
            toxicity_cutoff=design.toxicity_cutoff,
            efficacy_cutoff=design.efficacy_cutoff,
        )
        result: BOIN12Decision = design.next_dose(
            n,
            t,
            e,
            current,
            efficacy_without_toxicity=efficacy_without_toxicity,
            eliminated=excluded,
        )
        return BOIN12TwoStageDecision(
            2, result.action, result.next_dose, result.admissible, _owned(excluded), posterior
        )
    return _stage1_decision(
        design,
        n.astype(float, copy=False),
        t.astype(float, copy=False),
        current,
        excluded,
    )


def _joint_grid(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("joint_probability must be real-valued")
    return _validated_joint_grid(raw)


def simulate_boin12_two_stage(
    design: BOIN12Design,
    joint_probability: ArrayLike,
    *,
    stage1_threshold: int,
    cohorts: int = 12,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> BOIN12TwoStageSimulation:
    """Simulate complete-outcome cohorts under the two-stage BOIN12 policy."""
    _validate_design(design)
    threshold = _threshold(stage1_threshold)
    grid = _joint_grid(joint_probability)
    doses = grid.shape[0]
    start = _start(start_dose, doses)
    settings = count([cohorts, cohort_size, trials], "simulation settings")
    if settings.shape != (3,) or np.any(settings < 1):
        raise ValueError("cohorts, cohort_size and trials must be positive integers")
    ncohort, size, repetitions = (int(value) for value in settings)
    if ncohort * size > 1000:
        raise ValueError("the trial may enroll at most 1000 patients")
    if (
        repetitions > 1_000_000
        or repetitions * doses > 2_000_000
        or repetitions * ncohort * doses > 2_000_000
    ):
        raise ValueError("require at most 1000000 trials and 2000000 trial-dose cells")

    # All settings and dimensions are checked before creating or consuming RNG.
    generator = np.random.default_rng(rng)
    patients = np.zeros((repetitions, doses), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    efficacies = np.zeros_like(patients)
    efficacy_without_toxicity = np.zeros_like(patients)
    eliminated = np.zeros((repetitions, doses), dtype=bool)
    selected_obd = np.zeros(repetitions, dtype=np.int64)
    selected_mtd = np.zeros(repetitions, dtype=np.int64)
    reasons = np.full(repetitions, "max_cohorts", dtype="U40")
    current = np.full(repetitions, start, dtype=np.int64)
    active = np.ones(repetitions, dtype=bool)
    transition = np.zeros(repetitions, dtype=np.int64)
    stage1_cohorts = np.zeros(repetitions, dtype=np.int64)
    stage2_cohorts = np.zeros(repetitions, dtype=np.int64)

    for cohort_index in range(1, ncohort + 1):
        for trial in np.flatnonzero(active):
            dose = int(current[trial]) - 1
            if np.any(patients[trial] >= threshold):
                stage2_cohorts[trial] += 1
            else:
                stage1_cohorts[trial] += 1
            cell = generator.multinomial(size, grid[dose])
            no_tox_eff, no_tox_no_eff, tox_eff, tox_no_eff = (int(x) for x in cell)
            patients[trial, dose] += size
            toxicities[trial, dose] += tox_eff + tox_no_eff
            efficacies[trial, dose] += no_tox_eff + tox_eff
            efficacy_without_toxicity[trial, dose] += no_tox_eff
            if transition[trial] == 0 and np.any(patients[trial] >= threshold):
                transition[trial] = cohort_index
            decision = boin12_two_stage_next_dose(
                design,
                patients[trial],
                toxicities[trial],
                efficacies[trial],
                int(current[trial]),
                stage1_threshold=threshold,
                efficacy_without_toxicity=efficacy_without_toxicity[trial],
                eliminated=eliminated[trial],
            )
            eliminated[trial] = decision.eliminated
            if decision.next_dose is None:
                reasons[trial] = decision.action
                active[trial] = False
            else:
                current[trial] = decision.next_dose

    for trial_index in range(repetitions):
        result = design.select_obd(
            patients[trial_index],
            toxicities[trial_index],
            efficacies[trial_index],
            efficacy_without_toxicity=efficacy_without_toxicity[trial_index],
            eliminated=eliminated[trial_index],
        )
        if result.obd is not None:
            selected_obd[trial_index] = result.obd
        if result.mtd is not None:
            selected_mtd[trial_index] = result.mtd

    def summary(selected: NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        frequency = np.bincount(selected, minlength=doses + 1).astype(float) / repetitions
        return frequency, np.sqrt(frequency * (1.0 - frequency) / repetitions)

    obd_probability, obd_mcse = summary(selected_obd)
    mtd_probability, mtd_mcse = summary(selected_mtd)
    return BOIN12TwoStageSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(efficacies),
        _owned(efficacy_without_toxicity),
        _owned(eliminated),
        _owned(selected_obd),
        _owned(selected_mtd),
        _owned(obd_probability),
        _owned(obd_mcse),
        _owned(mtd_probability),
        _owned(mtd_mcse),
        tuple(str(reason) for reason in reasons),
        _owned(transition),
        _owned(stage1_cohorts),
        _owned(stage2_cohorts),
    )
