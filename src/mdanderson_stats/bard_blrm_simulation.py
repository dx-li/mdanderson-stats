"""Streaming operating characteristics for stochastic BARD BF-BLRM trials."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from ._validation import scalar
from .bard_bf_boin_simulation import (
    BARDCategoryFrequency,
    BARDMeanEstimate,
    BARDProportion,
    BARDSelectionAccuracy,
    _accuracy,
    _dose_frequencies,
    _frequencies,
    _OnlineMean,
    _proportion,
)
from .bard_bf_boin_trial import BARDStageTwoDesign
from .bard_blrm_generation import BARDBLRMSimulationDesign
from .bard_blrm_stochastic import BARDBLRMStochasticTrial, run_bard_blrm_stochastic_trial
from .bard_response import BARDResponseModel

_MAX_TRIALS = 100_000
_MAX_PATIENT_WORK = 100_000_000
_MAX_EVALUATIONS = 20_000_000
_MAX_WORK = 500_000_000


@dataclass(frozen=True, slots=True)
class BARDBLRMSimulation:
    """Compact repeated-trial summaries; every valid trial stays in denominators."""

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
    fit_count: int
    likelihood_evaluations: int
    work_units: int
    max_pod_mcse: float | None
    max_ptt_mcse: float | None
    max_finite_split_rhat: float | None
    trials_with_undefined_or_infinite_rhat: int
    patient_work_bound: int
    max_patient_work: int


def _positive_int(value: float, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not 1 <= int(parsed) <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return int(parsed)


def _dose_label(value: float, name: str, n_doses: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer one-based dose label")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not 1 <= int(parsed) <= n_doses:
        raise ValueError(f"{name} must be an integer in 1..{n_doses}")
    return int(parsed)


def _diagnostics(
    trial: BARDBLRMStochasticTrial,
) -> tuple[float | None, float | None, float | None, bool]:
    max_ptt_mcse: float | None = None
    max_pod_mcse: float | None = None
    max_rhat: float | None = None
    invalid_rhat = False
    for step in trial.stage_one.trial.steps:
        snapshot = step.snapshot
        finite_ptt = snapshot.ptt_mcse[np.isfinite(snapshot.ptt_mcse)]
        finite_pod = snapshot.pod_mcse[np.isfinite(snapshot.pod_mcse)]
        if finite_ptt.size:
            current = float(np.max(finite_ptt))
            max_ptt_mcse = current if max_ptt_mcse is None else max(max_ptt_mcse, current)
        if finite_pod.size:
            current = float(np.max(finite_pod))
            max_pod_mcse = current if max_pod_mcse is None else max(max_pod_mcse, current)
        values = np.concatenate((snapshot.ptt_split_rhat, snapshot.pod_split_rhat))
        if np.any(~np.isfinite(values)):
            invalid_rhat = True
        finite_rhat = values[np.isfinite(values)]
        if finite_rhat.size:
            current = float(np.max(finite_rhat))
            max_rhat = current if max_rhat is None else max(max_rhat, current)
    return max_ptt_mcse, max_pod_mcse, max_rhat, invalid_rhat


def simulate_bard_blrm(
    design: BARDBLRMSimulationDesign,
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    stage_two: BARDStageTwoDesign,
    *,
    true_obd_noninferiority: int,
    true_obd_utility: int,
    trials: int = 100,
    scenario_label: str | None = None,
    joint_toxicity_response_probability: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
    max_patient_work: int = _MAX_PATIENT_WORK,
    max_total_evaluations: int = _MAX_EVALUATIONS,
    max_total_work: int = _MAX_WORK,
) -> BARDBLRMSimulation:
    """Run serial stochastic BF-BLRM trials with aggregate patient/fit budgets.

    Arrival-tape exhaustion is an error from the trial runner, never a valid
    truncated trial. Posterior diagnostics are summarized for inspection and
    do not remove trials or imply convergence.
    """
    if not isinstance(design, BARDBLRMSimulationDesign):
        raise TypeError("design must be BARDBLRMSimulationDesign")
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be BARDResponseModel")
    if not isinstance(stage_two, BARDStageTwoDesign):
        raise TypeError("stage_two must be BARDStageTwoDesign")
    if scenario_label is not None and (
        not isinstance(scenario_label, str)
        or not scenario_label.strip()
        or len(scenario_label) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in scenario_label)
    ):
        raise ValueError("scenario_label must be nonempty printable text of at most 128 characters")
    n_trials = _positive_int(trials, "trials", _MAX_TRIALS)
    max_patients = _positive_int(max_patient_work, "max_patient_work", _MAX_PATIENT_WORK)
    eval_budget = _positive_int(max_total_evaluations, "max_total_evaluations", _MAX_EVALUATIONS)
    work_budget = _positive_int(max_total_work, "max_total_work", _MAX_WORK)
    profiles = response_model.factor_profiles
    if (
        not isinstance(profiles, np.ndarray)
        or profiles.ndim != 2
        or not 1 <= profiles.shape[0] <= 100_000
        or not 1 <= profiles.shape[1] <= 5
        or profiles.size > 1_000_000
    ):
        raise ValueError("response_model profiles exceed supported dimensions")
    n_doses = int(np.asarray(design.doses).size)
    if not isinstance(
        response_model.population_response, np.ndarray
    ) or response_model.population_response.shape != (n_doses,):
        raise ValueError("response_model population response must match the dose grid")
    true_ni = _dose_label(true_obd_noninferiority, "true_obd_noninferiority", n_doses)
    true_util = _dose_label(true_obd_utility, "true_obd_utility", n_doses)

    # Preflight the upper bound before creating or advancing the random stream.
    patient_bound = n_trials * (design.max_arrivals + stage_two.total_target)
    if patient_bound > max_patients:
        raise ValueError(
            f"patient_work_bound {patient_bound} exceeds max_patient_work={max_patients}"
        )
    minimum_work = design.chains * design.draws * n_doses
    minimum_all_trials = n_trials * minimum_work
    if design.max_total_work < minimum_work:
        raise ValueError("per-trial work budget cannot cover the minimum initial BF-BLRM fit")
    if design.max_total_evaluations < 1:
        raise ValueError("per-trial evaluation budget must permit likelihood evaluation")
    if work_budget < minimum_all_trials:
        raise ValueError("aggregate work budget cannot cover one minimum BF-BLRM fit per trial")
    if eval_budget < 1:
        raise ValueError(
            "aggregate evaluation budget must permit at least one likelihood evaluation"
        )

    if isinstance(rng, np.random.Generator):
        generator = rng
    elif rng is None or (
        not isinstance(rng, (bool, np.bool_))
        and isinstance(rng, (int, np.integer))
        and 0 <= int(rng) <= 2**64 - 1
    ):
        generator = np.random.default_rng(None if rng is None else int(rng))
    else:
        raise ValueError("rng must be None, a nonnegative 64-bit integer, or Generator")

    enrollment = _OnlineMean()
    duration = _OnlineMean()
    arm_gap = _OnlineMean()
    factor_gaps = [_OnlineMean() for _ in range(profiles.shape[1])]
    status_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()
    ni_status_counts: Counter[str] = Counter()
    util_status_counts: Counter[str] = Counter()
    ni_by_dose = np.zeros(n_doses, dtype=np.int64)
    util_by_dose = np.zeros(n_doses, dtype=np.int64)
    ni_correct = util_correct = ni_selected = util_selected = 0
    no_pair = safety_rejected = empty_factor = arm_metric = factor_metric = 0
    total_fits = total_evaluations = total_work = 0
    max_ptt_mcse = max_pod_mcse = max_rhat = None
    invalid_rhat_trials = 0

    for _ in range(n_trials):
        remaining_evals = eval_budget - total_evaluations
        remaining_work = work_budget - total_work
        if remaining_work < minimum_work or remaining_evals < 1:
            raise RuntimeError("aggregate BF-BLRM budget exhausted before the next trial")
        trial_design = replace(
            design,
            max_total_evaluations=min(design.max_total_evaluations, remaining_evals),
            max_total_work=min(design.max_total_work, remaining_work),
        )
        trial = run_bard_blrm_stochastic_trial(
            trial_design,
            true_toxicity,
            response_model,
            stage_two,
            joint_toxicity_response_probability=joint_toxicity_response_probability,
            rng=generator,
        )
        status_counts[trial.status] += 1
        stop_counts[trial.stage_one_stop_reason] += 1
        ni_status_counts[trial.noninferiority_status] += 1
        util_status_counts[trial.utility_status] += 1
        enrollment.add(float(trial.total_sample_size))
        duration.add(float(trial.duration))
        selected_ni = trial.selected_dose_noninferiority
        selected_util = trial.selected_dose_utility
        if selected_ni is not None:
            ni_selected += 1
            ni_by_dose[selected_ni - 1] += 1
            ni_correct += selected_ni == true_ni
        if selected_util is not None:
            util_selected += 1
            util_by_dose[selected_util - 1] += 1
            util_correct += selected_util == true_util

        if trial.dose_pair is None:
            no_pair += 1
        elif trial.status == "dose_pair_safety_eliminated":
            safety_rejected += 1
        else:
            arms = np.asarray(trial.arm_history, dtype=np.int64)
            factor = np.asarray(trial.factor_history, dtype=np.int64)
            mask1, mask2 = arms == 1, arms == 2
            arm1, arm2 = int(mask1.sum()), int(mask2.sum())
            arm_metric += 1
            arm_gap.add(float(abs(arm1 - arm2)))
            if arm1 and arm2:
                factor_metric += 1
                for index, accumulator in enumerate(factor_gaps):
                    p1 = float(np.count_nonzero(factor[mask1, index] == 1) / arm1)
                    p2 = float(np.count_nonzero(factor[mask2, index] == 1) / arm2)
                    accumulator.add(abs(p1 - p2))
            else:
                empty_factor += 1

        total_fits += trial.fit_count
        total_evaluations += trial.likelihood_evaluations
        total_work += trial.work_units
        trial_ptt_mcse, trial_pod_mcse, trial_rhat, has_invalid = _diagnostics(trial)
        if trial_ptt_mcse is not None:
            max_ptt_mcse = (
                trial_ptt_mcse if max_ptt_mcse is None else max(max_ptt_mcse, trial_ptt_mcse)
            )
        if trial_pod_mcse is not None:
            max_pod_mcse = (
                trial_pod_mcse if max_pod_mcse is None else max(max_pod_mcse, trial_pod_mcse)
            )
        if trial_rhat is not None:
            max_rhat = trial_rhat if max_rhat is None else max(max_rhat, trial_rhat)
        invalid_rhat_trials += int(has_invalid)

    return BARDBLRMSimulation(
        scenario_label,
        n_trials,
        true_ni,
        true_util,
        enrollment.summary(),
        duration.summary(),
        _frequencies(status_counts, n_trials),
        _frequencies(stop_counts, n_trials),
        _frequencies(ni_status_counts, n_trials),
        _frequencies(util_status_counts, n_trials),
        _dose_frequencies(ni_by_dose, n_trials),
        _proportion(n_trials - ni_selected, n_trials),
        _dose_frequencies(util_by_dose, n_trials),
        _proportion(n_trials - util_selected, n_trials),
        _accuracy(ni_correct, ni_selected, n_trials),
        _accuracy(util_correct, util_selected, n_trials),
        no_pair,
        safety_rejected,
        empty_factor,
        arm_metric,
        factor_metric,
        arm_gap.summary(),
        tuple(item.summary() for item in factor_gaps),
        total_fits,
        total_evaluations,
        total_work,
        max_pod_mcse,
        max_ptt_mcse,
        max_rhat,
        invalid_rhat_trials,
        patient_bound,
        max_patients,
    )
