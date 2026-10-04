"""Streaming operating characteristics for repeated BARD BF-BOIN trials."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .bard_bf_boin_trial import BARDBFBOINTrialResult, BARDStageTwoDesign, run_bard_bf_boin_trial
from .bard_response import BARDResponseModel
from .bf_boin import BFBOINDesign

_MAX_TRIALS = 1_000_000
_MAX_PATIENT_WORK = 100_000_000


@dataclass(frozen=True, slots=True)
class BARDProportion:
    """A Bernoulli summary with its exact denominator and plug-in MCSE."""

    count: int
    denominator: int
    probability: float
    mcse: float


@dataclass(frozen=True, slots=True)
class BARDMeanEstimate:
    """Streaming mean with sample SD and Monte Carlo standard error."""

    count: int
    mean: float | None
    sample_sd: float | None
    mcse: float | None


@dataclass(frozen=True, slots=True)
class BARDCategoryFrequency:
    """A dose or status count divided by an explicit trial denominator."""

    category: int | str
    count: int
    denominator: int
    proportion: float
    mcse: float


@dataclass(frozen=True, slots=True)
class BARDSelectionAccuracy:
    """Unconditional correct-selection probability and selected-only accuracy."""

    correct_all_trials: BARDProportion
    correct_given_selection: BARDProportion | None
    selected_trials: int


@dataclass(frozen=True, slots=True)
class BARDBFBOINSimulation:
    """Compact repeated-trial operating-characteristic summary.

    Selection probabilities and unconditional correct-selection probabilities
    use all simulated trials. Arm-count imbalance includes every non-rejected
    pair; factor-proportion imbalance requires both combined eligible arms to
    be nonempty. The metric denominators and exclusions are reported separately.
    """

    scenario_label: str | None
    trials: int
    true_obd_noninferiority: int
    true_obd_utility: int
    mean_total_enrollment: BARDMeanEstimate
    mean_duration: BARDMeanEstimate
    status_frequency: tuple[BARDCategoryFrequency, ...]
    stage_one_stop_reason_frequency: tuple[BARDCategoryFrequency, ...]
    noninferiority_status_frequency: tuple[BARDCategoryFrequency, ...]
    utility_status_frequency: tuple[BARDCategoryFrequency, ...]
    noninferiority_selection_by_dose: tuple[BARDCategoryFrequency, ...]
    noninferiority_no_selection: BARDProportion
    utility_selection_by_dose: tuple[BARDCategoryFrequency, ...]
    utility_no_selection: BARDProportion
    noninferiority_accuracy: BARDSelectionAccuracy
    utility_accuracy: BARDSelectionAccuracy
    no_pair_trials: int
    safety_rejected_pair_trials: int
    empty_arm_factor_metric_trials: int
    arm_count_metric_trials: int
    factor_metric_trials: int
    mean_pair_arm_count_imbalance: BARDMeanEstimate
    mean_factor_level1_proportion_imbalance: tuple[BARDMeanEstimate, ...]


class _OnlineMean:
    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def summary(self) -> BARDMeanEstimate:
        if self.count == 0:
            return BARDMeanEstimate(0, None, None, None)
        if self.count == 1:
            return BARDMeanEstimate(1, self.mean, None, None)
        sd = sqrt(max(0.0, self.m2 / (self.count - 1)))
        return BARDMeanEstimate(self.count, self.mean, sd, sd / sqrt(self.count))


def _proportion(successes: int, denominator: int) -> BARDProportion:
    if denominator <= 0 or successes < 0 or successes > denominator:
        raise ValueError("proportion counts must satisfy 0 <= count <= denominator")
    probability = successes / denominator
    return BARDProportion(
        successes,
        denominator,
        probability,
        sqrt(probability * (1.0 - probability) / denominator),
    )


def _frequencies(counts: Counter[str], denominator: int) -> tuple[BARDCategoryFrequency, ...]:
    return tuple(
        BARDCategoryFrequency(
            category=key,
            count=value,
            denominator=denominator,
            proportion=value / denominator,
            mcse=sqrt((value / denominator) * (1 - value / denominator) / denominator),
        )
        for key, value in sorted(counts.items())
    )


def _dose_frequencies(
    counts: NDArray[np.int64], denominator: int
) -> tuple[BARDCategoryFrequency, ...]:
    return tuple(
        BARDCategoryFrequency(
            dose,
            int(value),
            denominator,
            float(value / denominator),
            sqrt(float(value / denominator) * (1 - float(value / denominator)) / denominator),
        )
        for dose, value in enumerate(counts, start=1)
    )


def _accuracy(correct: int, selected: int, trials: int) -> BARDSelectionAccuracy:
    return BARDSelectionAccuracy(
        correct_all_trials=_proportion(correct, trials),
        correct_given_selection=None if selected == 0 else _proportion(correct, selected),
        selected_trials=selected,
    )


def simulate_bard_bf_boin(
    design: BFBOINDesign,
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    stage_two: BARDStageTwoDesign,
    *,
    true_obd_noninferiority: int,
    true_obd_utility: int,
    trials: int = 1_000,
    scenario_label: str | None = None,
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
    max_patient_work: int = _MAX_PATIENT_WORK,
) -> BARDBFBOINSimulation:
    """Run serial trials and return bounded streaming OC summaries.

    The two true OBD indices are separate, caller-supplied one-based dose
    labels for the noninferiority and utility analyses. A single NumPy
    Generator is used serially: an integer or ``None`` initializes it; a
    supplied Generator is advanced in place. Patient ledgers are discarded
    after each trial and never collected into the returned result.
    """
    if not isinstance(design, BFBOINDesign):
        raise TypeError("design must be a BFBOINDesign")
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be a BARDResponseModel")
    if not isinstance(stage_two, BARDStageTwoDesign):
        raise TypeError("stage_two must be a BARDStageTwoDesign")
    if scenario_label is not None and (
        not isinstance(scenario_label, str)
        or not scenario_label.strip()
        or len(scenario_label) > 128
    ):
        raise ValueError("scenario_label must be a nonempty string of at most 128 characters")
    model_profiles = response_model.factor_profiles
    if (
        not isinstance(model_profiles, np.ndarray)
        or model_profiles.ndim != 2
        or not 1 <= model_profiles.shape[0] <= 100_000
        or not 1 <= model_profiles.shape[1] <= 5
        or model_profiles.size > 1_000_000
    ):
        raise ValueError("response_model profiles exceed supported dimension bounds")
    if (
        not isinstance(response_model.population_response, np.ndarray)
        or response_model.population_response.ndim != 1
        or not 2 <= response_model.population_response.size <= 100
    ):
        raise ValueError("response_model must contain 2..100 dose margins")
    n_doses = response_model.population_response.size
    if isinstance(true_toxicity, np.ndarray):
        if (
            true_toxicity.ndim != 1
            or not 2 <= true_toxicity.size <= 100
            or true_toxicity.dtype.kind not in "iuf"
        ):
            raise ValueError("true_toxicity must be a vector with 2..100 dose probabilities")
    elif isinstance(true_toxicity, (list, tuple)):
        if not 2 <= len(true_toxicity) <= 100 or any(
            isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_))
            for item in true_toxicity
        ):
            raise ValueError("true_toxicity must be a bounded one-dimensional real sequence")
    else:
        raise ValueError("true_toxicity must be a bounded one-dimensional sequence")
    probabilities = np.asarray(true_toxicity, dtype=np.float64)
    if (
        probabilities.shape != (n_doses,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError(
            "true_toxicity must match response_model doses with probabilities in [0,1]"
        )

    def truth_dose(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer one-based dose label")
        result = int(value)
        if not 1 <= result <= n_doses:
            raise ValueError(f"{name} is outside the response-model dose grid")
        return result

    true_ni = truth_dose(true_obd_noninferiority, "true_obd_noninferiority")
    true_utility = truth_dose(true_obd_utility, "true_obd_utility")
    if isinstance(trials, (bool, np.bool_)):
        raise ValueError("trials must be an integer count")
    repetition_value = scalar(trials, "trials")
    if repetition_value != int(repetition_value) or not 1 <= repetition_value <= _MAX_TRIALS:
        raise ValueError(f"trials must be an integer in 1..{_MAX_TRIALS}")
    repetition_count = int(repetition_value)
    if any(isinstance(value, (bool, np.bool_)) for value in (cohorts, cohort_size, start_dose)):
        raise ValueError("cohorts, cohort_size, and start_dose must be integer settings")
    cohort_value = scalar(cohorts, "cohorts")
    cohort_size_value = scalar(cohort_size, "cohort_size")
    start_value = scalar(start_dose, "start_dose")
    if (
        cohort_value != int(cohort_value)
        or cohort_value < 1
        or cohort_size_value != int(cohort_size_value)
        or cohort_size_value < 1
        or start_value != int(start_value)
        or not 1 <= start_value <= n_doses
    ):
        raise ValueError("cohorts/cohort_size must be positive integers and start_dose valid")
    cohorts_int, cohort_size_int, start_int = (
        int(cohort_value),
        int(cohort_size_value),
        int(start_value),
    )
    titration_allowance = 0
    if accelerated_titration:
        if titration_cap is not None and (
            isinstance(titration_cap, (bool, np.bool_))
            or not isinstance(titration_cap, (int, np.integer))
            or not start_int <= int(titration_cap) <= n_doses
        ):
            raise ValueError("titration_cap must be a valid dose index")
        cap = n_doses if titration_cap is None else int(titration_cap)
        titration_allowance = cap - start_int + 1 + cohort_size_int
        if true_grade2 is None or grade2_assessment_delay is None:
            raise ValueError("accelerated titration requires grade-2 probabilities and delay")
    elif (
        titration_cap is not None or true_grade2 is not None or grade2_assessment_delay is not None
    ):
        raise ValueError("titration settings require accelerated_titration=True")
    stage_one_bound = cohorts_int * cohort_size_int + n_doses * design.n_cap + titration_allowance
    if stage_one_bound > 1_000:
        raise ValueError("a stage-one trial may enroll at most 1000 patients")
    maximum_trial_patients = stage_one_bound + stage_two.total_target
    if isinstance(max_patient_work, (bool, np.bool_)):
        raise ValueError("max_patient_work must be an integer bound")
    max_patient_work_value = scalar(max_patient_work, "max_patient_work")
    if (
        max_patient_work_value != int(max_patient_work_value)
        or not 1 <= max_patient_work_value <= _MAX_PATIENT_WORK
    ):
        raise ValueError(f"max_patient_work must be an integer in 1..{_MAX_PATIENT_WORK}")
    total_patient_work_bound = repetition_count * maximum_trial_patients
    if total_patient_work_bound > int(max_patient_work_value):
        raise ValueError(
            "requested patient-work bound "
            f"{total_patient_work_bound} exceeds max_patient_work={int(max_patient_work_value)}"
        )
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    statuses: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    ni_statuses: Counter[str] = Counter()
    utility_statuses: Counter[str] = Counter()
    ni_dose_counts = np.zeros(n_doses, dtype=np.int64)
    utility_dose_counts = np.zeros(n_doses, dtype=np.int64)
    ni_selected = utility_selected = ni_correct = utility_correct = 0
    no_pair = safety_rejected = empty_arm = arm_count_n = factor_n = 0
    total_n_summary, duration_summary, arm_imbalance_summary = (
        _OnlineMean(),
        _OnlineMean(),
        _OnlineMean(),
    )
    factor_imbalance_summaries = [
        _OnlineMean() for _ in range(response_model.factor_profiles.shape[1])
    ]

    for _ in range(repetition_count):
        result: BARDBFBOINTrialResult = run_bard_bf_boin_trial(
            design,
            probabilities,
            response_model,
            stage_two,
            cohorts=cohorts_int,
            cohort_size=cohort_size_int,
            start_dose=start_int,
            stage_one_accrual_rate=stage_one_accrual_rate,
            dlt_window=dlt_window,
            stage_one_arrival_distribution=stage_one_arrival_distribution,
            joint_toxicity_response_probability=joint_toxicity_response_probability,
            expand_after_escalation=expand_after_escalation,
            accelerated_titration=accelerated_titration,
            titration_cap=titration_cap,
            true_grade2=true_grade2,
            grade2_assessment_delay=grade2_assessment_delay,
            rng=generator,
        )
        statuses[result.status] += 1
        stop_reasons[result.stage_one_stop_reason] += 1
        ni_statuses[result.noninferiority_status] += 1
        utility_statuses[result.utility_status] += 1
        total_n_summary.add(float(result.dose_history.size))
        duration_summary.add(float(result.duration))
        if result.selected_dose_noninferiority is None:
            pass
        else:
            ni_selected += 1
            ni_dose_counts[result.selected_dose_noninferiority - 1] += 1
            ni_correct += result.selected_dose_noninferiority == true_ni
        if result.selected_dose_utility is None:
            pass
        else:
            utility_selected += 1
            utility_dose_counts[result.selected_dose_utility - 1] += 1
            utility_correct += result.selected_dose_utility == true_utility

        if result.dose_pair is None:
            no_pair += 1
            continue
        if result.status == "dose_pair_safety_eliminated":
            safety_rejected += 1
            continue
        arm_n = np.asarray(result.outcome_counts, dtype=np.int64).sum(axis=1)
        if arm_n.shape != (2,) or np.any(arm_n < 0):
            raise RuntimeError("trial output has malformed final pair-arm counts")
        arm_imbalance_summary.add(float(abs(int(arm_n[0]) - int(arm_n[1]))))
        arm_count_n += 1
        if np.any(arm_n == 0):
            empty_arm += 1
            continue
        n_stage_one = len(result.stage_one.assigned_history[0])
        eligible_mask = np.asarray(result.stage_one_eligible, dtype=bool)
        if eligible_mask.shape != (n_stage_one,):
            raise RuntimeError("stage-one carryover mask is not aligned with patient histories")
        factors = np.asarray(result.factor_history, dtype=np.int64)
        doses = np.asarray(result.dose_history, dtype=np.int64)
        if factors.ndim != 2 or factors.shape[1] != len(factor_imbalance_summaries):
            raise RuntimeError("trial factor history does not match modeled factors")
        n_stage_two = len(result.stage_two_patients)
        if factors.shape[0] != n_stage_one + n_stage_two or doses.shape != (factors.shape[0],):
            raise RuntimeError("combined stage factor and dose ledgers are not aligned")
        included = np.concatenate((eligible_mask, np.ones(n_stage_two, dtype=bool)))
        included &= np.isin(doses, result.dose_pair)
        included_factors, included_doses = factors[included], doses[included]
        if included_factors.shape[0] != int(arm_n.sum()):
            raise RuntimeError("patient-factor ledger does not reconcile with pair outcome counts")
        arm1 = included_doses == result.dose_pair[0]
        arm2 = included_doses == result.dose_pair[1]
        for factor_index, summary in enumerate(factor_imbalance_summaries):
            proportion1 = float(np.mean(included_factors[arm1, factor_index] == 1))
            proportion2 = float(np.mean(included_factors[arm2, factor_index] == 1))
            summary.add(abs(proportion1 - proportion2))
        factor_n += 1

    return BARDBFBOINSimulation(
        scenario_label=scenario_label,
        trials=repetition_count,
        true_obd_noninferiority=true_ni,
        true_obd_utility=true_utility,
        mean_total_enrollment=total_n_summary.summary(),
        mean_duration=duration_summary.summary(),
        status_frequency=_frequencies(statuses, repetition_count),
        stage_one_stop_reason_frequency=_frequencies(stop_reasons, repetition_count),
        noninferiority_status_frequency=_frequencies(ni_statuses, repetition_count),
        utility_status_frequency=_frequencies(utility_statuses, repetition_count),
        noninferiority_selection_by_dose=_dose_frequencies(ni_dose_counts, repetition_count),
        noninferiority_no_selection=_proportion(repetition_count - ni_selected, repetition_count),
        utility_selection_by_dose=_dose_frequencies(utility_dose_counts, repetition_count),
        utility_no_selection=_proportion(repetition_count - utility_selected, repetition_count),
        noninferiority_accuracy=_accuracy(ni_correct, ni_selected, repetition_count),
        utility_accuracy=_accuracy(utility_correct, utility_selected, repetition_count),
        no_pair_trials=no_pair,
        safety_rejected_pair_trials=safety_rejected,
        empty_arm_factor_metric_trials=empty_arm,
        arm_count_metric_trials=arm_count_n,
        factor_metric_trials=factor_n,
        mean_pair_arm_count_imbalance=arm_imbalance_summary.summary(),
        mean_factor_level1_proportion_imbalance=tuple(
            item.summary() for item in factor_imbalance_summaries
        ),
    )
