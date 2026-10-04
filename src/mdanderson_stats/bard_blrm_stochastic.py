"""Generated-outcome, two-stage BARD BF-BLRM trial simulation.

This module composes the established BF-BLRM stage-one calendar, shared BARD
outcome generator, two-stage minimization continuation, and final OBD rules.
It does not introduce a second posterior, escalation, minimization, or OBD
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bard import BARDSelectionResult, bard_select_obd
from .bard_bf_boin_trial import BARDStageTwoDesign
from .bard_blrm_generation import (
    BARDBLRMSimulationDesign,
    BARDGeneratedBLRMStageOne,
    _generate_bard_blrm_outcome_tapes,
    simulate_bard_blrm_stage_one,
)
from .bard_blrm_trial import BARDBLRMTrial
from .bard_integrated import BARDBLRMStage2Result, continue_bard_trial
from .bard_response import BARDResponseModel
from .bf_boin_simulation import _validate_bard_response_truth

_MAX_STAGE_TWO_CANDIDATES = 2_000
_MAX_STAGE_TWO_OUTCOME_CELLS = 200_000
_MAX_CONTINUATION_FACTOR_CELLS = 100_000
_MAX_CONTINUATION_WORK = 50_000_000


def _readonly(value: ArrayLike, dtype: type | np.dtype | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _count_joint(counts: NDArray[np.int64], arm: int, dlt: bool, response: bool) -> None:
    column = 2 if dlt and response else 0 if dlt else 3 if response else 1
    counts[arm - 1, column] += 1


@dataclass(frozen=True)
class BARDBLRMStochasticStageTwoPatient:
    """Generated stage-two patient with allocation and assessment ledger."""

    patient_index: int
    candidate_index: int
    arrival: float
    dose: int
    arm: int
    profile_index: int
    factors: NDArray[np.int64]
    dlt: bool
    response: bool
    dlt_assessment: float
    response_assessment: float
    response_probability: float
    conditional_response_probability: float
    allocation_seed: int
    allocation_scores: NDArray[np.float64]
    allocation_probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class BARDBLRMStochasticTrial:
    """Immutable two-stage BF-BLRM trial and complete generated-patient ledger."""

    stage_one: BARDGeneratedBLRMStageOne
    stage_two: BARDBLRMStage2Result | None
    status: str
    stage_one_stop_reason: str
    dose_pair: tuple[int, int] | None
    stage_one_eligible: NDArray[np.bool_]
    stage_one_carryover: int
    requested_total_target: int
    required_new_enrollment: int
    stage_two_enrollment: int
    shortfall: int
    stage_one_counts: NDArray[np.int64]
    stage_two_counts: NDArray[np.int64]
    outcome_counts: NDArray[np.int64]
    stage_two_patients: tuple[BARDBLRMStochasticStageTwoPatient, ...]
    final_noninferiority: BARDSelectionResult | None
    final_utility: BARDSelectionResult | None
    selected_dose_noninferiority: int | None
    selected_dose_utility: int | None
    noninferiority_status: str
    utility_status: str
    factor_history: NDArray[np.int64]
    dose_history: NDArray[np.int64]
    dlt_history: NDArray[np.bool_]
    response_history: NDArray[np.bool_]
    profile_index_history: NDArray[np.int64]
    response_probability_history: NDArray[np.float64]
    sampled_response_probability_history: NDArray[np.float64]
    arrival_history: NDArray[np.float64]
    arm_history: NDArray[np.int64]
    stage_two_start_time: float
    duration: float
    seed: int | None

    @property
    def total_sample_size(self) -> int:
        return int(self.dose_history.size)

    @property
    def fit_count(self) -> int:
        return self.stage_one.trial.fit_count

    @property
    def likelihood_evaluations(self) -> int:
        return self.stage_one.trial.likelihood_evaluations

    @property
    def work_units(self) -> int:
        return self.stage_one.trial.work_units


def run_bard_blrm_stochastic_trial(
    design: BARDBLRMSimulationDesign,
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    stage_two: BARDStageTwoDesign,
    *,
    joint_toxicity_response_probability: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> BARDBLRMStochasticTrial:
    """Simulate one generated-outcome BF-BLRM trial through final OBD selection.

    Stage two begins after complete stage-one follow-up. The shared outcome
    generator creates the eligible stage-two profile/outcome tape before the
    existing continuation consumes it for minimization allocations; this
    ordering is an explicit Python random-stream convention. Generated
    outcomes and timing are not claimed to reproduce the native app's random
    stream or calendar.
    """
    if not isinstance(design, BARDBLRMSimulationDesign):
        raise TypeError("design must be a BARDBLRMSimulationDesign")
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be a BARDResponseModel")
    if not isinstance(stage_two, BARDStageTwoDesign):
        raise TypeError("stage_two must be a BARDStageTwoDesign")

    # Cross-component/static validation precedes every random draw.
    toxicity_shape = np.shape(true_toxicity)
    if toxicity_shape != np.shape(response_model.population_response) or len(toxicity_shape) != 1:
        raise ValueError("true_toxicity must match the response model dose grid")
    if toxicity_shape[0] > 100 or np.iscomplexobj(true_toxicity):
        raise ValueError("true_toxicity must be a bounded real dose vector")
    toxicity = np.asarray(true_toxicity, dtype=np.float64)
    if not np.isfinite(toxicity).all() or np.any((toxicity < 0) | (toxicity > 1)):
        raise ValueError("true_toxicity must contain probabilities in [0,1]")
    model_fields = (
        response_model.population_response,
        response_model.factor_profiles,
        response_model.profile_probabilities,
        response_model.response_odds_ratios,
        response_model.intercepts,
        response_model.conditional_probabilities,
        response_model.marginal_residuals,
    )
    if any(not isinstance(value, np.ndarray) for value in model_fields):
        raise ValueError("response_model fields must come from bard_response_model")
    profiles = response_model.factor_profiles
    profile_weights = response_model.profile_probabilities
    if (
        profiles.ndim != 2
        or not 1 <= profiles.shape[1] <= 5
        or profiles.shape[0] < 1
        or profiles.shape[0] > 100_000
        or profiles.shape[0] * profiles.shape[1] > 500_000
        or profile_weights.shape != (profiles.shape[0],)
    ):
        raise ValueError("response model has unsupported profile dimensions")
    if np.shape(stage_two.eligible_profiles) != (profiles.shape[0],):
        raise ValueError("stage-two eligible_profiles must match response-model profiles")
    eligible_profiles = np.asarray(stage_two.eligible_profiles, dtype=bool)
    n_factors = profiles.shape[1]
    balanced_factors = (
        tuple(range(n_factors))
        if stage_two.balanced_factors is None
        else stage_two.balanced_factors
    )
    if not balanced_factors or max(balanced_factors) >= n_factors:
        raise ValueError("balanced_factors must select existing response-model columns")
    dose_pair = stage_two.dose_pair
    if dose_pair is not None and cast(tuple[int, int], dose_pair)[1] > toxicity.size:
        raise ValueError("stage-two dose_pair exceeds the BF-BLRM dose grid")
    stage_profile_weights = (
        profile_weights
        if stage_two.stage_two_profile_probabilities is None
        else np.asarray(stage_two.stage_two_profile_probabilities, dtype=np.float64)
    )
    if stage_profile_weights.shape != (profiles.shape[0],):
        raise ValueError("stage-two profile weights must match response-model profiles")
    if (
        not np.isfinite(stage_profile_weights).all()
        or np.any(stage_profile_weights < 0)
        or not np.isclose(stage_profile_weights.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("stage-two profile weights must be probabilities summing to one")
    conditioned_weights = np.where(eligible_profiles, stage_profile_weights, 0.0)
    eligible_mass = float(conditioned_weights.sum())
    if not np.isfinite(eligible_mass) or eligible_mass <= 0:
        raise ValueError("eligible profiles must have positive stage-two probability mass")
    conditioned_weights /= eligible_mass

    stage_two_joint = (
        joint_toxicity_response_probability
        if stage_two.stage_two_joint_toxicity_response_probability is None
        else stage_two.stage_two_joint_toxicity_response_probability
    )
    # Validate a stage-two q override before the stage-one random stream starts.
    _validate_bard_response_truth(
        response_model,
        np.asarray(response_model.population_response, dtype=np.float64),
        toxicity,
        stage_two_joint,
    )
    max_new = int(stage_two.total_target)
    if max_new > _MAX_STAGE_TWO_CANDIDATES:
        raise ValueError(
            f"stage-two total_target cannot exceed {_MAX_STAGE_TWO_CANDIDATES} generated arrivals"
        )
    if max_new * toxicity.size > _MAX_STAGE_TWO_OUTCOME_CELLS:
        raise ValueError("stage-two potential-outcome tape exceeds 200000 cells")
    if (design.max_arrivals + max_new) * len(balanced_factors) > _MAX_CONTINUATION_FACTOR_CELLS:
        raise ValueError("combined minimization factor history exceeds 100000 cells")
    worst_work = len(balanced_factors) * (
        design.max_arrivals * max_new + max_new * (max_new - 1) // 2
    )
    if worst_work > _MAX_CONTINUATION_WORK:
        raise ValueError("stage-two target exceeds the minimization work budget")
    stage_two_rate = float(stage_two.stage_two_accrual_rate)
    largest_gap_mean = (
        2.0 if stage_two.arrival_distribution == "uniform" else 1.0
    ) / stage_two_rate
    if not np.isfinite(largest_gap_mean) or largest_gap_mean <= 0:
        raise ValueError("stage-two accrual rate makes the arrival interval unrepresentable")
    with np.errstate(over="ignore", invalid="ignore"):
        stage_two_time_scale = largest_gap_mean * max_new
    if not np.isfinite(stage_two_time_scale):
        raise ValueError("stage-two target and accrual rate exceed the finite calendar range")
    # Validate the two final-analysis contracts before stage-one simulation.
    for method in ("noninferiority", "utility"):
        bard_select_obd(
            np.ones((2, 4), dtype=np.int64),
            prior=stage_two.prior,
            safety_weights=stage_two.safety_weights,
            toxicity_limit=stage_two.toxicity_limit,
            efficacy_limit=stage_two.efficacy_limit,
            safety_cutoff=stage_two.safety_cutoff,
            efficacy_cutoff=stage_two.efficacy_cutoff,
            method=method,
            utilities=stage_two.utilities,
            margin=stage_two.margin,
            tie_arm=stage_two.tie_arm,
        )

    # Keep one Generator across both phases. Explicit integer seeds and the
    # generated seed case are replayable; caller-owned Generator states advance
    # in place and cannot be reconstructed from a seed.
    if isinstance(rng, np.random.Generator):
        generator = rng
        captured_seed: int | None = None
    elif rng is None:
        captured_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        generator = np.random.default_rng(captured_seed)
    elif (
        isinstance(rng, (bool, np.bool_))
        or not isinstance(rng, (int, np.integer))
        or int(rng) < 0
        or int(rng) > 2**64 - 1
    ):
        raise ValueError("rng must be None, a nonnegative 64-bit seed, or a Generator")
    else:
        captured_seed = int(rng)
        generator = np.random.default_rng(captured_seed)

    generated = simulate_bard_blrm_stage_one(
        design,
        toxicity,
        response_model,
        joint_toxicity_response_probability=joint_toxicity_response_probability,
        rng=generator,
    )
    stage_one = generated.trial
    if stage_one.stop_reason == "arrival_schedule_exhausted":
        raise RuntimeError(
            "stage-one arrival schedule exhausted before protocol completion; increase max_arrivals"
        )

    if stage_one.selected_mtd is None or stage_one.all_overdose_detected:
        return _no_stage_two_result(
            generated,
            "stage_one_no_mtd",
            None,
            np.zeros((2, 4), dtype=np.int64),
            np.zeros(len(stage_one.patients), dtype=bool),
            0,
            stage_two,
            captured_seed,
        )
    pair_value = stage_two.dose_pair
    pair: tuple[int, int] | None = None if pair_value is None else cast(tuple[int, int], pair_value)
    if pair is None:
        if stage_one.selected_mtd < 2:
            return _no_stage_two_result(
                generated,
                "no_adjacent_lower_dose",
                None,
                np.zeros((2, 4), dtype=np.int64),
                np.zeros(len(stage_one.patients), dtype=bool),
                0,
                stage_two,
                captured_seed,
            )
        pair = (stage_one.selected_mtd - 1, stage_one.selected_mtd)
    if not all(bool(stage_one.final_selection.safe[dose - 1]) for dose in pair):
        return _no_stage_two_result(
            generated,
            "dose_pair_safety_eliminated",
            pair,
            np.zeros((2, 4), dtype=np.int64),
            np.zeros(len(stage_one.patients), dtype=bool),
            0,
            stage_two,
            captured_seed,
        )

    tapes = generated.outcome_tapes
    stage_factor_rows = np.asarray(
        [tapes.factor_profiles[patient.arrival_index] for patient in stage_one.patients],
        dtype=np.int64,
    ).reshape((-1, n_factors))
    stage_profile_indices = np.asarray(
        [tapes.profile_indices[patient.arrival_index] for patient in stage_one.patients],
        dtype=np.int64,
    )
    stage_doses = np.asarray([patient.dose for patient in stage_one.patients], dtype=np.int64)
    stage_dlt = np.asarray([patient.dlt for patient in stage_one.patients], dtype=bool)
    stage_response = np.asarray([patient.response for patient in stage_one.patients], dtype=bool)
    stage_arrivals = np.asarray([patient.arrival_time for patient in stage_one.patients])
    stage_marginal = np.asarray(
        [tapes.response_probabilities[p.arrival_index, p.dose - 1] for p in stage_one.patients],
        dtype=np.float64,
    )
    stage_conditional = np.asarray(
        [
            tapes.sampled_response_probabilities[p.arrival_index, p.dose - 1]
            for p in stage_one.patients
        ],
        dtype=np.float64,
    )

    eligible = np.zeros(len(stage_one.patients), dtype=bool)
    stage_one_counts = np.zeros((2, 4), dtype=np.int64)
    stage_one_arms = np.zeros(len(stage_one.patients), dtype=np.int64)
    for i, (dose, profile_index) in enumerate(zip(stage_doses, stage_profile_indices, strict=True)):
        if bool(eligible_profiles[profile_index]) and int(dose) in pair:
            eligible[i] = True
            arm = pair.index(int(dose)) + 1
            stage_one_arms[i] = arm
            _count_joint(stage_one_counts, arm, bool(stage_dlt[i]), bool(stage_response[i]))
    carryover = int(np.count_nonzero(eligible))
    requested_target = int(stage_two.total_target)
    required_new = max(0, requested_target - carryover)
    effective_target = max(requested_target, carryover)
    start_time = float(stage_one.final_time)
    stage_two_tapes = (
        _generate_bard_blrm_outcome_tapes(
            toxicity,
            response_model,
            arrivals=required_new,
            accrual_rate=stage_two.stage_two_accrual_rate,
            arrival_distribution=stage_two.arrival_distribution,
            dlt_window=design.dlt_window,
            joint_toxicity_response_probability=stage_two_joint,
            profile_probabilities=conditioned_weights,
            rng=generator,
            first_arrival_time=start_time,
        )
        if required_new
        else None
    )

    candidate_factors = (
        np.asarray(stage_two_tapes.factor_profiles[:, balanced_factors], dtype=np.int64)
        if stage_two_tapes is not None
        else np.empty((0, len(balanced_factors)), dtype=np.int64)
    )
    stage_factors_balanced = stage_factor_rows[:, balanced_factors]
    pair_columns = np.asarray(pair, dtype=np.int64) - 1
    candidate_toxicity = (
        np.asarray(stage_two_tapes.potential_toxicities[:, pair_columns], dtype=np.int64)
        if stage_two_tapes is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    candidate_response = (
        np.asarray(stage_two_tapes.potential_responses[:, pair_columns], dtype=np.int64)
        if stage_two_tapes is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    continuation = continue_bard_trial(
        stage_one,
        dose_pair=pair,
        stage_one_eligible=eligible,
        stage_one_factors=stage_factors_balanced,
        candidate_factors=candidate_factors,
        potential_toxicities=candidate_toxicity,
        potential_responses=candidate_response,
        total_target=effective_target,
        prior=stage_two.prior,
        safety_weights=stage_two.safety_weights,
        allocation_probability=stage_two.allocation_probability,
        tie_probability=stage_two.tie_probability,
        toxicity_limit=stage_two.toxicity_limit,
        efficacy_limit=stage_two.efficacy_limit,
        safety_cutoff=stage_two.safety_cutoff,
        efficacy_cutoff=stage_two.efficacy_cutoff,
        method="utility",
        utilities=stage_two.utilities,
        margin=stage_two.margin,
        tie_arm=stage_two.tie_arm,
        rng=generator,
    )
    if continuation.status != "completed" or continuation.final_selection is None:
        raise RuntimeError(f"stage-two continuation did not complete: {continuation.status}")
    stage_two_counts = np.asarray(continuation.stage_two_counts, dtype=np.int64)
    counts = np.asarray(continuation.outcome_counts, dtype=np.int64)
    utility = continuation.final_selection
    if np.all(counts.sum(axis=1) > 0):
        noninferiority = bard_select_obd(
            counts,
            prior=stage_two.prior,
            safety_weights=stage_two.safety_weights,
            toxicity_limit=stage_two.toxicity_limit,
            efficacy_limit=stage_two.efficacy_limit,
            safety_cutoff=stage_two.safety_cutoff,
            efficacy_cutoff=stage_two.efficacy_cutoff,
            method="noninferiority",
            utilities=stage_two.utilities,
            margin=stage_two.margin,
            tie_arm=stage_two.tie_arm,
        )
        ni_status = "completed"
    else:
        noninferiority = None
        ni_status = "insufficient_observations_both_arms"

    new_patients: list[BARDBLRMStochasticStageTwoPatient] = []
    assert stage_two_tapes is not None or required_new == 0
    stage_two_start = start_time
    duration_end = stage_one.final_time
    if stage_two_tapes is not None:
        for patient in continuation.patients:
            j = patient.candidate_index
            arm_column = pair.index(patient.dose)
            arrival = float(stage_two_tapes.arrival_times[j])
            dlt_delay = float(stage_two_tapes.dlt_assessment_delays[j, patient.dose - 1])
            response_delay = float(stage_two_tapes.response_assessment_delays[j, patient.dose - 1])
            dlt_assessment = arrival + dlt_delay
            response_assessment = arrival + response_delay
            if (
                not np.isfinite(dlt_assessment)
                or not np.isfinite(response_assessment)
                or (dlt_delay > 0 and dlt_assessment <= arrival)
                or (response_delay > 0 and response_assessment <= arrival)
            ):
                raise ArithmeticError(
                    "positive stage-two assessment delay is not representable at its calendar time"
                )
            new_patients.append(
                BARDBLRMStochasticStageTwoPatient(
                    patient_index=len(stage_one.patients) + j,
                    candidate_index=j,
                    arrival=arrival,
                    dose=patient.dose,
                    arm=arm_column + 1,
                    profile_index=int(stage_two_tapes.profile_indices[j]),
                    factors=_readonly(stage_two_tapes.factor_profiles[j], np.int64),
                    dlt=patient.toxicity,
                    response=patient.response,
                    dlt_assessment=dlt_assessment,
                    response_assessment=response_assessment,
                    response_probability=float(
                        stage_two_tapes.response_probabilities[j, patient.dose - 1]
                    ),
                    conditional_response_probability=float(
                        stage_two_tapes.sampled_response_probabilities[j, patient.dose - 1]
                    ),
                    allocation_seed=patient.allocation_seed,
                    allocation_scores=_readonly(patient.scores, np.float64),
                    allocation_probabilities=_readonly(patient.probabilities, np.float64),
                )
            )
            duration_end = max(duration_end, dlt_assessment, response_assessment)

    all_factors = np.vstack(
        (
            stage_factor_rows,
            np.vstack([p.factors for p in new_patients])
            if new_patients
            else np.empty((0, n_factors), dtype=np.int64),
        )
    )
    all_doses = np.concatenate(
        (stage_doses, np.asarray([p.dose for p in new_patients], dtype=np.int64))
    )
    all_dlt = np.concatenate((stage_dlt, np.asarray([p.dlt for p in new_patients], dtype=bool)))
    all_response = np.concatenate(
        (stage_response, np.asarray([p.response for p in new_patients], dtype=bool))
    )
    all_profiles = np.concatenate(
        (stage_profile_indices, np.asarray([p.profile_index for p in new_patients], dtype=np.int64))
    )
    all_marginal = np.concatenate(
        (
            stage_marginal,
            np.asarray([p.response_probability for p in new_patients], dtype=np.float64),
        )
    )
    all_conditional = np.concatenate(
        (
            stage_conditional,
            np.asarray(
                [p.conditional_response_probability for p in new_patients], dtype=np.float64
            ),
        )
    )
    all_arrivals = np.concatenate(
        (stage_arrivals, np.asarray([p.arrival for p in new_patients], dtype=np.float64))
    )
    all_arms = np.concatenate(
        (stage_one_arms, np.asarray([p.arm for p in new_patients], dtype=np.int64))
    )
    status = "carryover_exceeds_target" if carryover > requested_target else "completed"
    selected_noninferiority_dose = (
        None
        if noninferiority is None or noninferiority.selected_arm is None
        else pair[noninferiority.selected_arm - 1]
    )
    selected_utility_dose = None if utility.selected_arm is None else pair[utility.selected_arm - 1]
    return BARDBLRMStochasticTrial(
        generated,
        continuation,
        status,
        stage_one.stop_reason,
        pair,
        _readonly(eligible, bool),
        carryover,
        requested_target,
        required_new,
        len(new_patients),
        required_new - len(new_patients),
        _readonly(stage_one_counts, np.int64),
        _readonly(stage_two_counts, np.int64),
        _readonly(counts, np.int64),
        tuple(new_patients),
        noninferiority,
        utility,
        selected_noninferiority_dose,
        selected_utility_dose,
        ni_status,
        "completed",
        _readonly(all_factors, np.int64),
        _readonly(all_doses, np.int64),
        _readonly(all_dlt, bool),
        _readonly(all_response, bool),
        _readonly(all_profiles, np.int64),
        _readonly(all_marginal, np.float64),
        _readonly(all_conditional, np.float64),
        _readonly(all_arrivals, np.float64),
        _readonly(all_arms, np.int64),
        stage_two_start,
        max(0.0, duration_end - stage_one.start_time),
        captured_seed,
    )


def _no_stage_two_result(
    generated: BARDGeneratedBLRMStageOne,
    status: str,
    pair: tuple[int, int] | None,
    stage_one_counts: NDArray[np.int64],
    eligible: NDArray[np.bool_],
    carryover: int,
    stage_two: BARDStageTwoDesign,
    seed: int | None,
) -> BARDBLRMStochasticTrial:
    """Build a complete no-stage-two result without fabricating an OBD."""
    trial: BARDBLRMTrial = generated.trial
    tapes = generated.outcome_tapes
    factors = np.asarray(
        [tapes.factor_profiles[p.arrival_index] for p in trial.patients], dtype=np.int64
    ).reshape((-1, tapes.factor_profiles.shape[1]))
    profiles = np.asarray(
        [tapes.profile_indices[p.arrival_index] for p in trial.patients], dtype=np.int64
    )
    marginal = np.asarray(
        [tapes.response_probabilities[p.arrival_index, p.dose - 1] for p in trial.patients],
        dtype=np.float64,
    )
    conditional = np.asarray(
        [tapes.sampled_response_probabilities[p.arrival_index, p.dose - 1] for p in trial.patients],
        dtype=np.float64,
    )
    doses = np.asarray([p.dose for p in trial.patients], dtype=np.int64)
    dlt = np.asarray([p.dlt for p in trial.patients], dtype=bool)
    response = np.asarray([p.response for p in trial.patients], dtype=bool)
    arrivals = np.asarray([p.arrival_time for p in trial.patients], dtype=np.float64)
    empty_counts = np.zeros((2, 4), dtype=np.int64)
    return BARDBLRMStochasticTrial(
        generated,
        None,
        status,
        trial.stop_reason,
        pair,
        _readonly(eligible, bool),
        carryover,
        stage_two.total_target,
        max(0, stage_two.total_target - carryover),
        0,
        max(0, stage_two.total_target - carryover),
        _readonly(stage_one_counts, np.int64),
        _readonly(empty_counts, np.int64),
        _readonly(stage_one_counts, np.int64),
        (),
        None,
        None,
        None,
        None,
        "stage_two_not_run",
        "stage_two_not_run",
        _readonly(factors, np.int64),
        _readonly(doses, np.int64),
        _readonly(dlt, bool),
        _readonly(response, bool),
        _readonly(profiles, np.int64),
        _readonly(marginal, np.float64),
        _readonly(conditional, np.float64),
        _readonly(arrivals, np.float64),
        _readonly(np.zeros(len(doses), dtype=np.int64), np.int64),
        trial.final_time,
        trial.duration,
        seed,
    )
