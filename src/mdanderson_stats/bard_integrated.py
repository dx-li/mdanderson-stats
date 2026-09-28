"""Complete-outcome stage-two continuation for BARD BF-BLRM trials.

This module composes the stage-one BF-BLRM replay with the published
two-arm Pocock-Simon allocation and final OBD helpers. Stage-two candidates
are supplied in arrival order with completed potential binary outcomes. The
paper does not specify a stage-two assessment-time model or interim stopping
rule, so this is not a delayed-outcome calendar replay.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite, scalar
from .bard import BARDSelectionResult, bard_minimization, bard_select_obd
from .bard_blrm_trial import BARDBLRMTrial

_MAX_CANDIDATES = 100_000
_MAX_FACTOR_CELLS = 100_000
_MAX_WORK = 50_000_000


def _readonly(value: ArrayLike, dtype: np.dtype | type | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _binary_matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.bool_]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real binary values")
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must have shape {shape} and contain binary values")
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all() or np.any((converted != 0) & (converted != 1)):
        raise ValueError(f"{name} must contain finite zero/one values")
    return converted.astype(bool)


def _factor_matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.int64]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real categorical values")
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must have shape {shape} and contain integer categories")
    converted = np.asarray(array, dtype=np.float64)
    if (
        not np.isfinite(converted).all()
        or np.any(converted < 1)
        or np.any(converted != np.floor(converted))
        or np.any(converted >= float(2**63))
    ):
        raise ValueError(f"{name} must contain positive finite integer categories")
    result = converted.astype(np.int64)
    if result.shape[1] < 1 or result.shape[1] > 5:
        raise ValueError(f"{name} must contain 1..5 categorical factors")
    return result


@dataclass(frozen=True)
class BARDBLRMStage2Patient:
    """One new Stage-2 assignment and its completed potential outcome."""

    candidate_index: int
    dose: int
    toxicity: bool
    response: bool
    allocation_seed: int
    scores: NDArray[np.float64]
    probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class BARDBLRMStage2Result:
    """Immutable Stage-2 continuation ledger and final OBD result."""

    status: str
    dose_pair: tuple[int, int]
    stage_one_stop_reason: str
    total_target: int
    stage_one_eligible: NDArray[np.bool_]
    stage_one_included_indices: tuple[int, ...]
    stage_one_excluded_ineligible_indices: tuple[int, ...]
    stage_one_excluded_other_dose_indices: tuple[int, ...]
    stage_one_counts: NDArray[np.int64]
    stage_two_counts: NDArray[np.int64]
    outcome_counts: NDArray[np.int64]
    stage_one_carryover: int
    required_new_enrollment: int
    stage_two_enrollment: int
    shortfall: int
    candidates_examined: int
    patients: tuple[BARDBLRMStage2Patient, ...]
    final_selection: BARDSelectionResult | None

    @property
    def selected_arm(self) -> int | None:
        return None if self.final_selection is None else self.final_selection.selected_arm


def continue_bard_trial(
    stage_one: BARDBLRMTrial,
    *,
    dose_pair: ArrayLike,
    stage_one_eligible: ArrayLike,
    stage_one_factors: ArrayLike,
    candidate_factors: ArrayLike,
    potential_toxicities: ArrayLike,
    potential_responses: ArrayLike,
    total_target: int,
    prior: ArrayLike,
    safety_weights: ArrayLike,
    allocation_probability: float,
    tie_probability: float,
    toxicity_limit: float,
    efficacy_limit: float,
    safety_cutoff: float,
    efficacy_cutoff: float,
    method: str,
    utilities: ArrayLike,
    margin: float,
    tie_arm: int,
    rng: np.random.Generator,
    max_work: int = _MAX_WORK,
) -> BARDBLRMStage2Result:
    """Continue a completed stage-one trial through two-arm Stage 2.

    ``total_target`` is the inclusive target total across the selected arms,
    including eligible stage-one carryover, as defined in the paper. The
    candidate rows represent eligible new patients in arrival order; their
    potential toxicity/response outcomes are columns for ``dose_pair``.
    Only the outcome for the assigned arm is counted.

    This function requires a two-dose transition pair. It does not require the
    higher dose to equal the stage-one MTD or the lower dose to be adjacent:
    the paper says the pair is selected from the totality of stage-one evidence
    and describes MTD plus its adjacent lower dose as the usual app default.
    There are no stage-two calendar delays or interim stop rules in the source;
    OBD is selected only after the inclusive target is reached.
    """
    if not isinstance(stage_one, BARDBLRMTrial):
        raise TypeError("stage_one must be a BARDBLRMTrial")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")

    patient_count = len(stage_one.patients)
    pair_shape = np.shape(dose_pair)
    eligible_shape = np.shape(stage_one_eligible)
    stage_factor_shape = np.shape(stage_one_factors)
    candidate_factor_shape = np.shape(candidate_factors)
    tox_shape = np.shape(potential_toxicities)
    response_shape = np.shape(potential_responses)
    if pair_shape != (2,) or np.iscomplexobj(dose_pair):
        raise ValueError("dose_pair must contain two real one-based dose indices")
    if eligible_shape != (patient_count,):
        raise ValueError("stage_one_eligible must have one Boolean value per stage-one patient")
    if np.iscomplexobj(stage_one_eligible):
        raise ValueError("stage_one_eligible must be Boolean")
    if len(candidate_factor_shape) != 2 or not 1 <= candidate_factor_shape[1] <= 5:
        raise ValueError("candidate_factors must have 1..5 columns")
    n_candidates, n_factors = candidate_factor_shape
    if n_candidates > _MAX_CANDIDATES:
        raise ValueError(f"candidate tape exceeds {_MAX_CANDIDATES} rows")
    if stage_factor_shape != (patient_count, n_factors):
        raise ValueError(
            "stage_one_factors must match stage-one patients and candidate factor count"
        )
    if tox_shape != (n_candidates, 2) or response_shape != (n_candidates, 2):
        raise ValueError("potential outcome matrices must have shape (candidate_count, 2)")
    if (patient_count + n_candidates) * n_factors > _MAX_FACTOR_CELLS:
        raise ValueError("combined factor history exceeds 100000 cells")

    target_value = scalar(total_target, "total_target")
    work_limit_value = scalar(max_work, "max_work")
    if target_value != int(target_value) or target_value < 1 or target_value > _MAX_CANDIDATES:
        raise ValueError(f"total_target must be an integer in 1..{_MAX_CANDIDATES}")
    if work_limit_value != int(work_limit_value) or not 1 <= work_limit_value <= _MAX_WORK:
        raise ValueError(f"max_work must be an integer in 1..{_MAX_WORK}")
    target = int(target_value)
    allocation_p = scalar(allocation_probability, "allocation_probability")
    tie_p = scalar(tie_probability, "tie_probability")
    if not 0.5 <= allocation_p <= 1 or not 0 <= tie_p <= 1:
        raise ValueError("allocation_probability must be in [.5,1] and tie_probability in [0,1]")
    pair_raw = finite(dose_pair, "dose_pair")
    if np.any(pair_raw != np.floor(pair_raw)):
        raise ValueError("dose_pair entries must be integer dose indices")
    pair = (int(pair_raw[0]), int(pair_raw[1]))
    if not 1 <= pair[0] < pair[1] <= stage_one.final_assigned.size:
        raise ValueError(
            "dose_pair must be ordered valid one-based indices in the stage-one dose grid"
        )

    eligible_raw = np.asarray(stage_one_eligible)
    if eligible_raw.dtype.kind != "b":
        if eligible_raw.dtype.kind not in "iu" or np.any((eligible_raw != 0) & (eligible_raw != 1)):
            raise ValueError("stage_one_eligible must contain Boolean values")
    eligible = np.asarray(eligible_raw, dtype=bool)
    stage_factors = _factor_matrix(
        stage_one_factors, (patient_count, n_factors), "stage_one_factors"
    )
    candidate_factor = _factor_matrix(
        candidate_factors, (n_candidates, n_factors), "candidate_factors"
    )
    toxic = _binary_matrix(potential_toxicities, (n_candidates, 2), "potential_toxicities")
    response = _binary_matrix(potential_responses, (n_candidates, 2), "potential_responses")

    stage_one_counts = np.zeros((2, 4), dtype=np.int64)
    included: list[int] = []
    excluded_ineligible: list[int] = []
    excluded_other: list[int] = []
    history_arms: list[int] = []
    history_factors: list[NDArray[np.int64]] = []
    for index, patient in enumerate(stage_one.patients):
        if not eligible[index]:
            excluded_ineligible.append(index)
            continue
        if patient.dose not in pair:
            excluded_other.append(index)
            continue
        included.append(index)
        arm = pair.index(patient.dose) + 1
        category = 0 if patient.dlt and not patient.response else 1
        if patient.dlt and patient.response:
            category = 2
        elif not patient.dlt and patient.response:
            category = 3
        stage_one_counts[arm - 1, category] += 1
        history_arms.append(arm)
        history_factors.append(stage_factors[index])

    carryover = len(included)
    if target < carryover:
        raise ValueError(
            "total_target cannot be smaller than mandatory eligible stage-one carryover"
        )
    required = target - carryover
    stage_one_terminal = stage_one.selected_mtd is None or stage_one.all_overdose_detected
    planned_assignments = 0 if stage_one_terminal else min(required, n_candidates)
    work_required = n_factors * (
        planned_assignments * carryover + planned_assignments * (planned_assignments - 1) // 2
    )
    if work_required > work_limit_value:
        raise ValueError("max_work cannot cover the requested stage-two minimization history")

    # Validate all final OBD settings before any allocation randomness is used.
    # Positive dummy counts also validate the noninferiority branch, which
    # requires observed responses in both arms.
    bard_select_obd(
        np.ones((2, 4), dtype=np.int64),
        prior=prior,
        safety_weights=safety_weights,
        toxicity_limit=toxicity_limit,
        efficacy_limit=efficacy_limit,
        safety_cutoff=safety_cutoff,
        efficacy_cutoff=efficacy_cutoff,
        method=method,
        utilities=utilities,
        margin=margin,
        tie_arm=tie_arm,
    )

    stage_two_counts = np.zeros((2, 4), dtype=np.int64)
    assignments: list[BARDBLRMStage2Patient] = []
    selection: BARDSelectionResult | None = None
    candidates_examined = 0

    # A permanently unsafe or MTD-less stage-one replay is not reopened by
    # Stage 2. Keep the carryover ledger for audit, but do not randomize.
    if stage_one_terminal:
        status = "stage_one_no_mtd"
    else:
        for index in range(min(required, n_candidates)):
            candidates_examined = index + 1
            seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            allocation = bard_minimization(
                history_arms,
                np.asarray(history_factors, dtype=np.int64).reshape((-1, n_factors)),
                candidate_factor[index],
                probability=allocation_p,
                tie_probability=tie_p,
                seed=seed,
            )
            arm = allocation.assigned_arm
            tox = bool(toxic[index, arm - 1])
            resp = bool(response[index, arm - 1])
            category = 0 if tox and not resp else 1
            if tox and resp:
                category = 2
            elif not tox and resp:
                category = 3
            stage_two_counts[arm - 1, category] += 1
            history_arms.append(arm)
            history_factors.append(candidate_factor[index])
            assignments.append(
                BARDBLRMStage2Patient(
                    candidate_index=index,
                    dose=pair[arm - 1],
                    toxicity=tox,
                    response=resp,
                    allocation_seed=seed,
                    scores=_readonly(allocation.scores),
                    probabilities=_readonly(allocation.probabilities),
                )
            )
        complete = len(assignments) == required
        if complete:
            if method in ("noninferiority", "noninferior") and np.any(
                (stage_one_counts + stage_two_counts).sum(axis=1) == 0
            ):
                status = "completed_selection_unavailable"
            else:
                status = "completed"
                selection = bard_select_obd(
                    stage_one_counts + stage_two_counts,
                    prior=prior,
                    safety_weights=safety_weights,
                    toxicity_limit=toxicity_limit,
                    efficacy_limit=efficacy_limit,
                    safety_cutoff=safety_cutoff,
                    efficacy_cutoff=efficacy_cutoff,
                    method=method,
                    utilities=utilities,
                    margin=margin,
                    tie_arm=tie_arm,
                )
        else:
            status = "candidate_tape_exhausted"

    outcome_counts = stage_one_counts + stage_two_counts
    enrollment = len(assignments)
    return BARDBLRMStage2Result(
        status=status,
        dose_pair=pair,
        stage_one_stop_reason=stage_one.stop_reason,
        total_target=target,
        stage_one_eligible=_readonly(eligible, bool),
        stage_one_included_indices=tuple(included),
        stage_one_excluded_ineligible_indices=tuple(excluded_ineligible),
        stage_one_excluded_other_dose_indices=tuple(excluded_other),
        stage_one_counts=_readonly(stage_one_counts, np.int64),
        stage_two_counts=_readonly(stage_two_counts, np.int64),
        outcome_counts=_readonly(outcome_counts, np.int64),
        stage_one_carryover=carryover,
        required_new_enrollment=required,
        stage_two_enrollment=enrollment,
        shortfall=required - enrollment,
        candidates_examined=candidates_examined,
        patients=tuple(assignments),
        final_selection=selection,
    )
