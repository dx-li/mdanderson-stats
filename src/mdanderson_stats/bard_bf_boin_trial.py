"""Single-trial BARD orchestration for BF-BOIN followed by stage two.

Stage one is delegated to :func:`simulate_bf_boin`; this module only carries
its completed patient ledger into BARD's two-arm minimization and OBD rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, scalar
from .bard import BARDSelectionResult, bard_minimization, bard_select_obd
from .bard_response import BARDResponseModel
from .bf_boin import BFBOINDesign
from .bf_boin_simulation import (
    BFBOINSimulation,
    _bard_conditional_response_probability,
    _validate_bard_response_truth,
    _weibull_endpoint,
    simulate_bf_boin,
)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
_MAX_STAGE_TWO_PATIENTS = 1_000


def _readonly(value: ArrayLike, dtype: type | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _bounded_vector(value: ArrayLike, name: str, maximum: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= maximum or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real array")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= maximum or any(
            isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_)) for item in value
        ):
            raise ValueError(f"{name} must be a bounded one-dimensional real sequence")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional real sequence")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    return result


def _bool_vector(value: ArrayLike, name: str, maximum: int) -> BoolArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= maximum or value.dtype.kind != "b":
            raise ValueError(f"{name} must be a bounded one-dimensional Boolean array")
        return _readonly(value, bool)
    if isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= maximum or any(
            not isinstance(item, (bool, np.bool_)) for item in value
        ):
            raise ValueError(f"{name} must contain bounded Boolean values")
        return _readonly(value, bool)
    raise ValueError(f"{name} must be a bounded Boolean sequence")


def _small_vector(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real array with shape {shape}")
    elif isinstance(value, (list, tuple)):
        if len(shape) == 1:
            if len(value) != shape[0] or any(
                isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_))
                for item in value
            ):
                raise ValueError(f"{name} must be a real sequence with shape {shape}")
        elif len(shape) == 2:
            if len(value) != shape[0] or any(
                not isinstance(row, (list, tuple, np.ndarray))
                or (isinstance(row, np.ndarray) and row.ndim != 1)
                or len(row) != shape[1]
                or any(
                    isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_))
                    for item in row
                )
                for row in value
            ):
                raise ValueError(f"{name} must be a real sequence with shape {shape}")
        else:
            raise ValueError("unsupported bounded vector shape")
    else:
        raise ValueError(f"{name} must be a real sequence with shape {shape}")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values with shape {shape}")
    return result


def _bounded_probability_matrix(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.size > 1_000_000 or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a bounded two-dimensional real array")
        rows, cols = value.shape
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if not 1 <= rows <= 100:
            raise ValueError(f"{name} row count must be in 1..100")
        first = value[0]
        if not isinstance(first, (list, tuple, np.ndarray)) or (
            isinstance(first, np.ndarray) and first.ndim != 1
        ):
            raise ValueError(f"{name} must be a rectangular matrix")
        cols = len(first)
        if not 1 <= cols <= 100_000 or rows * cols > 1_000_000:
            raise ValueError(f"{name} exceeds the bounded matrix dimensions")
        for row in value:
            if (
                not isinstance(row, (list, tuple, np.ndarray))
                or (isinstance(row, np.ndarray) and (row.ndim != 1 or row.dtype.kind not in "iuf"))
                or len(row) != cols
                or any(
                    isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_))
                    for item in row
                )
            ):
                raise ValueError(f"{name} must contain only real scalar cells in a rectangle")
    else:
        raise ValueError(f"{name} must be a bounded two-dimensional real matrix")
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (rows, cols)
        or not np.isfinite(array).all()
        or np.any((array < 0) | (array > 1))
    ):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return array


def _pair(value: ArrayLike | None, n_doses: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape != (2,) or value.dtype.kind not in "iu":
            raise ValueError("dose_pair must contain two integer one-based indices")
        result: tuple[int, int] = (int(value[0]), int(value[1]))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
            raise ValueError("dose_pair must contain two integer one-based indices")
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError("dose_pair must contain two integer one-based indices")
    if not 1 <= result[0] < result[1] <= n_doses:
        raise ValueError("dose_pair must be two ordered valid one-based dose indices")
    return result


@dataclass(frozen=True, slots=True)
class BARDStageTwoDesign:
    """Explicit BARD stage-two eligibility, allocation, and OBD policy.

    ``eligible_profiles`` is aligned to ``BARDResponseModel.factor_profiles``
    and defines eligibility for stage two. Stage-one eligible patients on the
    selected pair are mandatory carryover; new stage-two profile weights are
    conditioned and renormalized over this same eligible set.
    ``dose_pair=None`` requests the Python default of selected stage-one MTD
    plus its adjacent lower dose. Stage-two profile weights and joint endpoint
    probabilities, when supplied, do not recalibrate the response model.
    """

    total_target: int
    eligible_profiles: ArrayLike
    prior: ArrayLike
    safety_weights: ArrayLike
    toxicity_limit: float
    efficacy_limit: float
    safety_cutoff: float
    efficacy_cutoff: float
    utilities: ArrayLike
    margin: float
    tie_arm: int
    stage_two_accrual_rate: float
    dose_pair: ArrayLike | None = None
    stage_two_profile_probabilities: ArrayLike | None = None
    stage_two_joint_toxicity_response_probability: ArrayLike | None = None
    allocation_probability: float = 0.95
    tie_probability: float = 0.5
    balanced_factors: tuple[int, ...] | None = None
    arrival_distribution: str = "uniform"

    def __post_init__(self) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (
                self.total_target,
                self.stage_two_accrual_rate,
                self.allocation_probability,
                self.tie_probability,
                self.margin,
                self.tie_arm,
            )
        ):
            raise ValueError("stage-two scalar settings cannot be Boolean")
        target = scalar(self.total_target, "total_target")
        rate = scalar(self.stage_two_accrual_rate, "stage_two_accrual_rate")
        allocation = scalar(self.allocation_probability, "allocation_probability")
        tie = scalar(self.tie_probability, "tie_probability")
        margin = scalar(self.margin, "margin")
        arm = scalar(self.tie_arm, "tie_arm")
        if target != int(target) or not 1 <= target <= _MAX_STAGE_TWO_PATIENTS:
            raise ValueError(f"total_target must be an integer in 1..{_MAX_STAGE_TWO_PATIENTS}")
        if rate <= 0 or allocation < 0.5 or allocation > 1 or not 0 <= tie <= 1:
            raise ValueError("invalid stage-two accrual or allocation probability")
        if margin < 0 or arm != int(arm) or int(arm) not in (1, 2):
            raise ValueError("margin must be nonnegative and tie_arm must be 1 or 2")
        if self.arrival_distribution not in ("uniform", "exponential"):
            raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
        eligible = _bool_vector(self.eligible_profiles, "eligible_profiles", 100_000)
        if isinstance(self.prior, np.ndarray):
            prior_shape = self.prior.shape
            prior_items = self.prior.tolist()
        elif isinstance(self.prior, (list, tuple)):
            prior_shape = None
            prior_items = list(self.prior)
        else:
            raise ValueError("prior must have shape (4,) or (2,4)")
        if prior_shape not in ((4,), (2, 4)):
            if not isinstance(self.prior, (list, tuple)) or not (
                len(prior_items) == 4
                and all(np.isscalar(v) for v in prior_items)
                or len(prior_items) == 2
                and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in prior_items)
            ):
                raise ValueError("prior must have shape (4,) or (2,4)")
        prior_shape = (
            (4,)
            if prior_shape is None and all(np.isscalar(v) for v in prior_items)
            else (2, 4)
            if prior_shape is None
            else prior_shape
        )
        prior = _readonly(_small_vector(self.prior, prior_shape, "prior"), np.float64)
        weights = _readonly(_small_vector(self.safety_weights, (2,), "safety_weights"), np.float64)
        utilities = _readonly(_small_vector(self.utilities, (4,), "utilities"), np.float64)
        for name, item in (
            ("toxicity_limit", self.toxicity_limit),
            ("efficacy_limit", self.efficacy_limit),
            ("safety_cutoff", self.safety_cutoff),
            ("efficacy_cutoff", self.efficacy_cutoff),
        ):
            number = scalar(item, name)
            if not 0 <= number <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, number)
        if self.balanced_factors is not None:
            if not isinstance(self.balanced_factors, tuple) or not self.balanced_factors:
                raise ValueError("balanced_factors must be a nonempty tuple of zero-based columns")
            if any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in self.balanced_factors
            ):
                raise ValueError("balanced_factors must contain nonnegative integer columns")
            if len(set(self.balanced_factors)) != len(self.balanced_factors):
                raise ValueError("balanced_factors cannot repeat columns")
        profile_weights = (
            None
            if self.stage_two_profile_probabilities is None
            else _readonly(
                _bounded_vector(
                    self.stage_two_profile_probabilities, "stage_two_profile_probabilities", 100_000
                )
            )
        )
        joint = self.stage_two_joint_toxicity_response_probability
        if joint is not None:
            joint = _readonly(
                _bounded_probability_matrix(joint, "stage_two_joint_toxicity_response_probability"),
                np.float64,
            )
        pair = _pair(self.dose_pair, 100)
        object.__setattr__(self, "total_target", int(target))
        object.__setattr__(self, "stage_two_accrual_rate", rate)
        object.__setattr__(self, "allocation_probability", allocation)
        object.__setattr__(self, "tie_probability", tie)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "tie_arm", int(arm))
        object.__setattr__(self, "eligible_profiles", eligible)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "safety_weights", weights)
        object.__setattr__(self, "utilities", utilities)
        object.__setattr__(self, "stage_two_profile_probabilities", profile_weights)
        object.__setattr__(self, "stage_two_joint_toxicity_response_probability", joint)
        object.__setattr__(self, "dose_pair", pair)


@dataclass(frozen=True, slots=True)
class BARDBFBOINStage2Patient:
    """One randomized stage-two patient's assignment and observed outcomes."""

    patient_index: int
    arrival: float
    dose: int
    arm: int
    profile_index: int
    factors: IntArray
    dlt: bool
    response: bool
    dlt_assessment: float
    response_assessment: float
    conditional_response_probability: float
    response_probability: float
    allocation_scores: FloatArray
    allocation_probabilities: FloatArray
    allocation_seed: int


@dataclass(frozen=True, slots=True)
class BARDStageOneSettings:
    """Stage-one calendar choices used by the complete trial replay."""

    cohorts: int
    cohort_size: int
    start_dose: int
    accrual_rate: float
    dlt_window: float
    arrival_distribution: str
    expand_after_escalation: bool
    accelerated_titration: bool
    titration_cap: int | None
    true_grade2: FloatArray | None
    grade2_assessment_delay: float | None
    joint_toxicity_response_probability: FloatArray | None


@dataclass(frozen=True, slots=True)
class BARDBFBOINTrialResult:
    """Immutable full patient ledger and both BARD final OBD analyses."""

    stage_one: BFBOINSimulation
    stage_one_design: BFBOINDesign
    true_toxicity: FloatArray
    response_model: BARDResponseModel
    stage_one_settings: BARDStageOneSettings
    stage_two_design: BARDStageTwoDesign
    rng_seed: int | None
    status: str
    stage_one_stop_reason: str
    dose_pair: tuple[int, int] | None
    stage_one_eligible: BoolArray
    stage_one_carryover: int
    required_new_enrollment: int
    stage_two_enrollment: int
    shortfall: int
    stage_one_counts: IntArray
    stage_two_counts: IntArray
    outcome_counts: IntArray
    stage_two_patients: tuple[BARDBFBOINStage2Patient, ...]
    final_noninferiority: BARDSelectionResult | None
    final_utility: BARDSelectionResult | None
    selected_dose_noninferiority: int | None
    selected_dose_utility: int | None
    noninferiority_status: str
    utility_status: str
    factor_history: IntArray
    dose_history: IntArray
    dlt_history: BoolArray
    response_history: BoolArray
    profile_index_history: IntArray
    response_probability_history: FloatArray
    sampled_response_probability_history: FloatArray
    arrival_history: FloatArray
    stage_two_start_time: float
    duration: float

    @property
    def total_sample_size(self) -> int:
        return int(self.dose_history.size)

    @property
    def stage_one_sample_size(self) -> int:
        return self.total_sample_size - self.stage_two_enrollment


def _count_joint(counts: NDArray[np.int64], arm: int, dlt: bool, response: bool) -> None:
    column = 2 if dlt and response else 0 if dlt else 3 if response else 1
    counts[arm - 1, column] += 1


def _one_count(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be in 1..{maximum}")
    return result


def run_bard_bf_boin_trial(
    design: BFBOINDesign,
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    stage_two: BARDStageTwoDesign,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    start_dose: int = 1,
    stage_one_accrual_rate: float = 1.0,
    dlt_window: float = 1.0,
    stage_one_arrival_distribution: str = "uniform",
    joint_toxicity_response_probability: ArrayLike | None = None,
    expand_after_escalation: bool = False,
    accelerated_titration: bool = False,
    titration_cap: int | None = None,
    true_grade2: ArrayLike | None = None,
    grade2_assessment_delay: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> BARDBFBOINTrialResult:
    """Run one BF-BOIN stage followed by BARD allocation and OBD selection.

    Stage two starts only after complete stage-one follow-up. Its patients
    arrive individually under the configured renewal process. DLT event times
    reuse the BF-BOIN Weibull calibration and response is assessed at the end
    of the DLT window. Those timing choices are explicit Python conventions.
    """
    if not isinstance(design, BFBOINDesign) or not isinstance(response_model, BARDResponseModel):
        raise TypeError("design and response_model must be BFBOINDesign and BARDResponseModel")
    if not isinstance(stage_two, BARDStageTwoDesign):
        raise TypeError("stage_two must be a BARDStageTwoDesign")
    toxicity = _bounded_vector(true_toxicity, "true_toxicity", 100)
    if toxicity.size < 2 or np.any((toxicity < 0) | (toxicity > 1)):
        raise ValueError("true_toxicity must contain 2..100 dose probabilities")
    profiles, model_weights, stage_one_response, stage_one_joint = _validate_bard_response_truth(
        response_model,
        response_model.population_response,
        toxicity,
        joint_toxicity_response_probability=joint_toxicity_response_probability,
    )
    if profiles is None or model_weights is None or stage_one_response is None:
        raise RuntimeError(
            "BARD response truth validation did not return the calibrated profile table"
        )
    stage_two_q_input = (
        joint_toxicity_response_probability
        if stage_two.stage_two_joint_toxicity_response_probability is None
        else stage_two.stage_two_joint_toxicity_response_probability
    )
    stage_two_response: FloatArray | None
    stage_two_joint: FloatArray | None
    if stage_two_q_input is joint_toxicity_response_probability:
        stage_two_response, stage_two_joint = stage_one_response, stage_one_joint
    else:
        _, _, stage_two_response, stage_two_joint = _validate_bard_response_truth(
            response_model,
            response_model.population_response,
            toxicity,
            joint_toxicity_response_probability=stage_two_q_input,
        )
    if stage_two_response is None:
        raise RuntimeError("BARD response truth validation did not return stage-two probabilities")
    n_profiles, n_factors = profiles.shape
    eligible_profile_mask = np.asarray(stage_two.eligible_profiles)
    if eligible_profile_mask.shape != (n_profiles,) or eligible_profile_mask.dtype.kind != "b":
        raise ValueError("eligible_profiles must align with response_model profiles")
    eligible_profile_flags = np.asarray(eligible_profile_mask, dtype=bool)
    balanced_factors = (
        tuple(range(n_factors))
        if stage_two.balanced_factors is None
        else stage_two.balanced_factors
    )
    if max(balanced_factors) >= n_factors:
        raise ValueError("balanced_factors references a missing factor column")
    profile_weights = (
        model_weights
        if stage_two.stage_two_profile_probabilities is None
        else _bounded_vector(
            stage_two.stage_two_profile_probabilities, "stage_two_profile_probabilities", n_profiles
        )
    )
    if profile_weights.shape != (n_profiles,) or np.any(profile_weights < 0):
        raise ValueError("stage-two profile weights must match profiles and be nonnegative")
    source_weight_sum = float(profile_weights.sum())
    if not np.isfinite(source_weight_sum) or abs(source_weight_sum - 1) > 1e-12:
        raise ValueError("stage-two profile weights must sum to one")
    profile_weights = np.where(eligible_profile_mask, profile_weights, 0.0)
    weight_sum = float(profile_weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("eligible profiles must have positive stage-two probability mass")
    profile_weights = profile_weights / weight_sum
    n_doses = toxicity.size
    pair = _pair(stage_two.dose_pair, n_doses)
    if np.any(~np.isfinite(stage_two_response)):
        raise ValueError("stage-two conditional response probabilities must be finite")
    for method in ("utility", "noninferiority"):
        # Validate the complete final-analysis contract before stage-one RNG use.
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
    rate = scalar(stage_two.stage_two_accrual_rate, "stage_two_accrual_rate")
    window = scalar(dlt_window, "dlt_window")
    if rate <= 0 or window <= 0:
        raise ValueError("accrual rates and dlt_window must be positive")
    cohort_count = _one_count(cohorts, "cohorts", 100_000)
    cohort_size_value = _one_count(cohort_size, "cohort_size", 100_000)
    start_dose_value = _one_count(start_dose, "start_dose", n_doses)
    stage_one_rate = scalar(stage_one_accrual_rate, "stage_one_accrual_rate")
    if stage_one_rate <= 0:
        raise ValueError("stage_one_accrual_rate must be positive")
    if stage_one_arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("stage_one_arrival_distribution must be 'uniform' or 'exponential'")
    if not isinstance(expand_after_escalation, (bool, np.bool_)) or not isinstance(
        accelerated_titration, (bool, np.bool_)
    ):
        raise ValueError("stage-one option flags must be Boolean")
    grade2_snapshot = (
        None
        if true_grade2 is None
        else _readonly(_bounded_vector(true_grade2, "true_grade2", n_doses))
    )
    grade2_delay_value = (
        None
        if grade2_assessment_delay is None
        else scalar(grade2_assessment_delay, "grade2_assessment_delay")
    )
    if grade2_delay_value is not None and grade2_delay_value <= 0:
        raise ValueError("grade2_assessment_delay must be positive")
    if isinstance(start_dose, (bool, np.bool_)):
        raise ValueError("start_dose must be an integer dose index")
    start_value = scalar(start_dose, "start_dose")
    if start_value != int(start_value) or not 1 <= start_value <= n_doses:
        raise ValueError("start_dose must be a valid integer dose index")
    if titration_cap is not None:
        if isinstance(titration_cap, (bool, np.bool_)):
            raise ValueError("titration_cap must be an integer dose")
        cap_value = scalar(titration_cap, "titration_cap")
        if cap_value != int(cap_value) or not start_value <= cap_value <= n_doses:
            raise ValueError("titration_cap must be a valid dose index")
        titration_cap = int(cap_value)
    if isinstance(rng, np.random.Generator):
        generator = rng
        rng_seed = None
    else:
        if rng is not None and (
            isinstance(rng, (bool, np.bool_))
            or not isinstance(rng, (int, np.integer))
            or int(rng) < 0
            or int(rng) > 2**64 - 1
        ):
            raise ValueError("rng must be None, a nonnegative 64-bit seed, or a Generator")
        rng_seed = (
            int(rng)
            if rng is not None
            else int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        )
        generator = np.random.default_rng(rng_seed)
    stage1_settings = BARDStageOneSettings(
        cohort_count,
        cohort_size_value,
        start_dose_value,
        stage_one_rate,
        window,
        stage_one_arrival_distribution,
        bool(expand_after_escalation),
        bool(accelerated_titration),
        titration_cap,
        grade2_snapshot,
        grade2_delay_value,
        None if stage_one_joint is None else _readonly(stage_one_joint),
    )
    stage1 = simulate_bf_boin(
        design,
        toxicity,
        response_model.population_response,
        cohorts=cohort_count,
        cohort_size=cohort_size_value,
        trials=1,
        start_dose=start_dose_value,
        accrual_rate=stage_one_rate,
        dlt_window=window,
        arrival_distribution=stage_one_arrival_distribution,
        expand_after_escalation=expand_after_escalation,
        accelerated_titration=accelerated_titration,
        titration_cap=titration_cap,
        true_grade2=true_grade2,
        grade2_assessment_delay=grade2_assessment_delay,
        joint_toxicity_response_probability=stage_one_joint,
        response_model=response_model,
        rng=generator,
    )
    if (
        stage1.profile_index_history is None
        or stage1.factor_history is None
        or stage1.response_probability_history is None
        or stage1.sampled_response_probability_history is None
        or stage1.eliminated_history is None
    ):
        raise RuntimeError("BF-BOIN response-model histories were not retained")
    selected_mtd = int(stage1.selected_dose[0])
    stop_reason = stage1.stop_reason[0]
    if selected_mtd == 0:
        return _make_no_stage_two_result(
            stage1,
            "stage_one_no_mtd",
            stop_reason,
            stage1.factor_history[0],
            stage1.assigned_history[0],
            stage1.dlt_history[0],
            stage1.response_history[0],
            stage1.arrival_history[0],
            np.zeros((2, 4), dtype=np.int64),
            stage_one_design=design,
            true_toxicity=toxicity,
            response_model=response_model,
            stage_one_settings=stage1_settings,
            stage_two_design=stage_two,
            rng_seed=rng_seed,
        )
    if pair is None and selected_mtd > 1:
        pair = (selected_mtd - 1, selected_mtd)
    if pair is None:
        status = "no_adjacent_lower_dose"
        empty = np.zeros((2, 4), dtype=np.int64)
        stage_profiles = stage1.factor_history[0]
        stage_doses = stage1.assigned_history[0]
        stage_dlt = stage1.dlt_history[0]
        stage_response = stage1.response_history[0]
        stage_arrivals = stage1.arrival_history[0]
        return _make_no_stage_two_result(
            stage1,
            status,
            stop_reason,
            stage_profiles,
            stage_doses,
            stage_dlt,
            stage_response,
            stage_arrivals,
            empty,
            stage_one_design=design,
            true_toxicity=toxicity,
            response_model=response_model,
            stage_one_settings=stage1_settings,
            stage_two_design=stage_two,
            rng_seed=rng_seed,
        )
    final_safety = design.select_mtd(stage1.patients[0], stage1.toxicities[0])
    ever_eliminated = np.asarray(stage1.eliminated_history[0], dtype=bool)
    unsafe = np.asarray(final_safety.eliminated, dtype=bool) | ever_eliminated
    if any(unsafe[dose - 1] for dose in pair):
        return _make_no_stage_two_result(
            stage1,
            "dose_pair_safety_eliminated",
            stop_reason,
            stage1.factor_history[0],
            stage1.assigned_history[0],
            stage1.dlt_history[0],
            stage1.response_history[0],
            stage1.arrival_history[0],
            np.zeros((2, 4), dtype=np.int64),
            pair,
            stage_one_design=design,
            true_toxicity=toxicity,
            response_model=response_model,
            stage_one_settings=stage1_settings,
            stage_two_design=stage_two,
            rng_seed=rng_seed,
        )
    stage1_profiles = stage1.factor_history[0]
    stage1_indices = stage1.profile_index_history[0]
    stage1_doses = stage1.assigned_history[0]
    stage1_dlt = stage1.dlt_history[0]
    stage1_response = stage1.response_history[0]
    stage1_arrivals = stage1.arrival_history[0]
    eligible = np.zeros(len(stage1_doses), dtype=bool)
    one_counts = np.zeros((2, 4), dtype=np.int64)
    history_arms: list[int] = []
    history_factors: list[NDArray[np.int64]] = []
    for index, (dose, profile_idx) in enumerate(zip(stage1_doses, stage1_indices, strict=True)):
        is_eligible = bool(eligible_profile_flags[int(profile_idx)])
        eligible[index] = is_eligible and int(dose) in pair
        if eligible[index]:
            arm = pair.index(int(dose)) + 1
            _count_joint(one_counts, arm, bool(stage1_dlt[index]), bool(stage1_response[index]))
            history_arms.append(arm)
            history_factors.append(stage1_profiles[index, list(balanced_factors)])
    carryover = len(history_arms)
    required = max(0, stage_two.total_target - carryover)
    status = "carryover_exceeds_target" if carryover > stage_two.total_target else "completed"
    two_counts = np.zeros((2, 4), dtype=np.int64)
    new_patients: list[BARDBFBOINStage2Patient] = []
    history_factor_array = np.asarray(history_factors, dtype=np.int64).reshape(
        -1, len(balanced_factors)
    )
    stage_two_start = float(stage1.trial_duration[0])
    enrollment_time = stage_two_start
    max_completion = enrollment_time
    endpoints = tuple(_weibull_endpoint(float(p), window) for p in toxicity)
    for patient_index in range(required):
        if patient_index == 0:
            arrival = enrollment_time
        else:
            gap = (
                generator.uniform(0, 2 / rate)
                if stage_two.arrival_distribution == "uniform"
                else generator.exponential(1 / rate)
            )
            arrival = enrollment_time + gap
            if not np.isfinite(arrival) or arrival <= enrollment_time:
                raise RuntimeError("stage-two arrival clock failed to advance")
            enrollment_time = arrival
        profile_index = int(generator.choice(n_profiles, p=profile_weights))
        factors = profiles[profile_index]
        allocation_seed = int(generator.integers(0, 2**63, dtype=np.int64))
        allocation = bard_minimization(
            history_arms,
            history_factor_array,
            factors[list(balanced_factors)],
            probability=stage_two.allocation_probability,
            tie_probability=stage_two.tie_probability,
            seed=allocation_seed,
        )
        arm = allocation.assigned_arm
        dose = pair[arm - 1]
        toxicity_probability = float(toxicity[dose - 1])
        shape, scale = endpoints[dose - 1]
        event_time = (
            np.inf
            if np.isinf(scale)
            else window / 2
            if scale == 0
            else scale * generator.weibull(shape)
        )
        dlt = bool(event_time <= window)
        marginal_response = float(stage_two_response[dose - 1, profile_index])
        joint = None if stage_two_joint is None else float(stage_two_joint[dose - 1, profile_index])
        conditional_response = _bard_conditional_response_probability(
            marginal_response, toxicity_probability, joint, dlt
        )
        if not 0 <= conditional_response <= 1 or not np.isfinite(conditional_response):
            raise ArithmeticError("stage-two conditional response probability is invalid")
        response = bool(generator.random() < conditional_response)
        dlt_assessment = arrival + min(event_time, window)
        response_assessment = arrival + window
        if not np.isfinite(response_assessment):
            raise RuntimeError("stage-two assessment time exceeds finite calendar range")
        patient = BARDBFBOINStage2Patient(
            len(stage1_doses) + patient_index,
            float(arrival),
            int(dose),
            int(arm),
            profile_index,
            _readonly(factors, np.int64),
            dlt,
            response,
            float(dlt_assessment),
            float(response_assessment),
            float(conditional_response),
            marginal_response,
            _readonly(allocation.scores, np.float64),
            _readonly(allocation.probabilities, np.float64),
            allocation_seed,
        )
        new_patients.append(patient)
        _count_joint(two_counts, arm, dlt, response)
        history_arms.append(arm)
        history_factor_array = np.vstack((history_factor_array, factors[list(balanced_factors)]))
        max_completion = max(max_completion, response_assessment)
    total_counts = one_counts + two_counts
    noninferiority = _select_if_both_arms(total_counts, stage_two, "noninferiority")
    ni_status = "completed" if noninferiority is not None else "insufficient_observations_both_arms"
    utility = bard_select_obd(
        total_counts,
        prior=stage_two.prior,
        safety_weights=stage_two.safety_weights,
        toxicity_limit=stage_two.toxicity_limit,
        efficacy_limit=stage_two.efficacy_limit,
        safety_cutoff=stage_two.safety_cutoff,
        efficacy_cutoff=stage_two.efficacy_cutoff,
        method="utility",
        utilities=stage_two.utilities,
        margin=stage_two.margin,
        tie_arm=stage_two.tie_arm,
    )
    factors_all = np.vstack(
        (
            stage1_profiles,
            np.vstack([p.factors for p in new_patients])
            if new_patients
            else np.empty((0, n_factors), dtype=np.int64),
        )
    )
    dose_all = np.concatenate(
        (stage1_doses, np.asarray([p.dose for p in new_patients], dtype=np.int64))
    )
    dlt_all = np.concatenate((stage1_dlt, np.asarray([p.dlt for p in new_patients], dtype=bool)))
    response_all = np.concatenate(
        (stage1_response, np.asarray([p.response for p in new_patients], dtype=bool))
    )
    arrival_all = np.concatenate(
        (stage1_arrivals, np.asarray([p.arrival for p in new_patients], dtype=np.float64))
    )
    profile_index_all = np.concatenate(
        (stage1_indices, np.asarray([p.profile_index for p in new_patients], dtype=np.int64))
    )
    marginal_probability_all = np.concatenate(
        (
            stage1.response_probability_history[0],
            np.asarray([p.response_probability for p in new_patients], dtype=np.float64),
        )
    )
    sampled_probability_all = np.concatenate(
        (
            stage1.sampled_response_probability_history[0],
            np.asarray(
                [p.conditional_response_probability for p in new_patients], dtype=np.float64
            ),
        )
    )
    return BARDBFBOINTrialResult(
        stage1,
        design,
        _readonly(toxicity),
        response_model,
        stage1_settings,
        stage_two,
        rng_seed,
        status,
        stop_reason,
        pair,
        _readonly(eligible, bool),
        carryover,
        required,
        len(new_patients),
        max(0, stage_two.total_target - carryover - len(new_patients)),
        _readonly(one_counts, np.int64),
        _readonly(two_counts, np.int64),
        _readonly(total_counts, np.int64),
        tuple(new_patients),
        noninferiority,
        utility,
        None
        if noninferiority is None or noninferiority.selected_arm is None
        else pair[noninferiority.selected_arm - 1],
        None if utility.selected_arm is None else pair[utility.selected_arm - 1],
        ni_status,
        "completed",
        _readonly(factors_all, np.int64),
        _readonly(dose_all, np.int64),
        _readonly(dlt_all, bool),
        _readonly(response_all, bool),
        _readonly(profile_index_all, np.int64),
        _readonly(marginal_probability_all, np.float64),
        _readonly(sampled_probability_all, np.float64),
        _readonly(arrival_all, np.float64),
        stage_two_start,
        float(max_completion),
    )


def _select_if_both_arms(
    counts: IntArray, design: BARDStageTwoDesign, method: str
) -> BARDSelectionResult | None:
    if np.any(counts.sum(axis=1) == 0):
        return None
    return bard_select_obd(
        counts,
        prior=design.prior,
        safety_weights=design.safety_weights,
        toxicity_limit=design.toxicity_limit,
        efficacy_limit=design.efficacy_limit,
        safety_cutoff=design.safety_cutoff,
        efficacy_cutoff=design.efficacy_cutoff,
        method=method,
        utilities=design.utilities,
        margin=design.margin,
        tie_arm=design.tie_arm,
    )


def _make_no_stage_two_result(
    stage1: BFBOINSimulation,
    status: str,
    stop_reason: str,
    factors: ArrayLike,
    doses: ArrayLike,
    dlt: ArrayLike,
    response: ArrayLike,
    arrivals: ArrayLike,
    counts: IntArray,
    pair: tuple[int, int] | None = None,
    *,
    stage_one_design: BFBOINDesign,
    true_toxicity: FloatArray,
    response_model: BARDResponseModel,
    stage_one_settings: BARDStageOneSettings,
    stage_two_design: BARDStageTwoDesign,
    rng_seed: int | None,
) -> BARDBFBOINTrialResult:
    empty_counts = _readonly(np.zeros((2, 4), dtype=np.int64), np.int64)
    factor_array = _readonly(factors, np.int64)
    dose_array = np.asarray(doses, dtype=np.int64)
    dlt_array = np.asarray(dlt, dtype=bool)
    response_array = np.asarray(response, dtype=bool)
    arrival_array = np.asarray(arrivals, dtype=np.float64)
    if any(
        value.ndim != 1 or value.size > 1_000
        for value in (dose_array, dlt_array, response_array, arrival_array)
    ):
        raise RuntimeError("stage-one patient history exceeded its validated bounds")
    if (
        stage1.profile_index_history is None
        or stage1.response_probability_history is None
        or stage1.sampled_response_probability_history is None
    ):
        raise RuntimeError("stage-one response-model histories are missing")
    profile_indices = stage1.profile_index_history[0]
    marginal_probabilities = stage1.response_probability_history[0]
    sampled_probabilities = stage1.sampled_response_probability_history[0]
    return BARDBFBOINTrialResult(
        stage1,
        stage_one_design,
        _readonly(true_toxicity),
        response_model,
        stage_one_settings,
        stage_two_design,
        rng_seed,
        status,
        stop_reason,
        pair,
        _readonly(np.zeros(dose_array.size, dtype=bool), bool),
        0,
        0,
        0,
        0,
        empty_counts,
        empty_counts,
        empty_counts,
        (),
        None,
        None,
        None,
        None,
        "stage_two_not_run",
        "stage_two_not_run",
        factor_array,
        _readonly(dose_array, np.int64),
        _readonly(dlt_array, bool),
        _readonly(response_array, bool),
        _readonly(profile_indices, np.int64),
        _readonly(marginal_probabilities, np.float64),
        _readonly(sampled_probabilities, np.float64),
        _readonly(arrival_array, np.float64),
        float(stage1.trial_duration[0]),
        float(stage1.trial_duration[0]),
    )
