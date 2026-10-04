import copy

import numpy as np
import pytest

from mdanderson_stats.bard_bf_boin_trial import BARDStageTwoDesign
from mdanderson_stats.bard_blrm import BARDLogisticPrior
from mdanderson_stats.bard_blrm_generation import BARDBLRMSimulationDesign
from mdanderson_stats.bard_blrm_simulation import simulate_bard_blrm
from mdanderson_stats.bard_blrm_stochastic import run_bard_blrm_stochastic_trial
from mdanderson_stats.bard_response import bard_response_model


def _inputs(*, prior: BARDLogisticPrior | None = None):
    response = bard_response_model(
        [0.20, 0.35, 0.50],
        [[1, 1], [1, 2], [2, 1], [2, 2]],
        [0.25] * 4,
        [[1.0, 1.5], [1.0, 0.8]],
    )
    design = BARDBLRMSimulationDesign(
        doses=[1.0, 2.0, 3.0],
        reference_dose=1.0,
        prior=prior or BARDLogisticPrior([-2.0, -0.6931471805599453], [0.0, 0.0]),
        target_interval=[0.20, 0.30],
        eta=0.45,
        cohort_size=3,
        max_escalation_patients=9,
        backfill_evaluable_cap=2,
        draws=8,
        warmup=0,
        chains=2,
        max_arrivals=40,
        arrival_distribution="uniform",
        accrual_rate=2.0,
        dlt_window=1.0,
        boundary_policy="stop",
    )
    stage_two = BARDStageTwoDesign(
        total_target=6,
        eligible_profiles=[True] * 4,
        prior=[1.0] * 4,
        safety_weights=[1.0, 1.0],
        toxicity_limit=0.30,
        efficacy_limit=0.20,
        safety_cutoff=0.99,
        efficacy_cutoff=0.99,
        utilities=[0.0, 30.0, 50.0, 100.0],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=2.0,
        dose_pair=(1, 2),
        balanced_factors=(0,),
    )
    return design, np.zeros(3), response, stage_two


def test_streaming_summary_matches_serial_full_trial_reducer():
    design, toxicity, response, stage_two = _inputs()
    summary = simulate_bard_blrm(
        design,
        toxicity,
        response,
        stage_two,
        true_obd_noninferiority=3,
        true_obd_utility=3,
        trials=2,
        rng=165,
    )
    generator = np.random.default_rng(165)
    trials = [
        run_bard_blrm_stochastic_trial(
            design,
            toxicity,
            response,
            stage_two,
            rng=generator,
        )
        for _ in range(2)
    ]
    sizes = np.asarray([trial.total_sample_size for trial in trials], dtype=float)
    durations = np.asarray([trial.duration for trial in trials], dtype=float)
    assert summary.mean_total_enrollment.mean == pytest.approx(float(sizes.mean()))
    assert summary.mean_duration.mean == pytest.approx(float(durations.mean()))
    assert summary.trials == 2
    assert sum(item.count for item in summary.status_frequency) == 2
    assert summary.noninferiority_accuracy.correct_all_trials.denominator == 2
    assert summary.utility_accuracy.correct_all_trials.denominator == 2
    assert (
        summary.no_pair_trials
        + summary.safety_rejected_pair_trials
        + summary.arm_count_metric_trials
        == 2
    )
    assert (
        summary.factor_metric_trials + summary.empty_arm_factor_metric_trials
        == summary.arm_count_metric_trials
    )
    assert all(
        item.count == summary.factor_metric_trials
        for item in summary.mean_factor_level1_proportion_imbalance
    )
    assert summary.no_pair_trials < summary.trials
    assert summary.arm_count_metric_trials > 0
    assert summary.factor_metric_trials > 0
    assert len(summary.mean_factor_level1_proportion_imbalance) == 2
    assert summary.mean_pair_arm_count_imbalance.mean is not None
    assert summary.mean_factor_level1_proportion_imbalance[0].mean is not None
    assert summary.mean_factor_level1_proportion_imbalance[1].mean is not None
    assert not hasattr(summary, "trial_results")

    expected_ni = [
        sum(trial.selected_dose_noninferiority == dose for trial in trials) for dose in range(1, 4)
    ]
    expected_util = [
        sum(trial.selected_dose_utility == dose for trial in trials) for dose in range(1, 4)
    ]
    assert [item.count for item in summary.noninferiority_selection_by_dose] == expected_ni
    assert [item.count for item in summary.utility_selection_by_dose] == expected_util
    pair_trials = [
        trial
        for trial in trials
        if trial.dose_pair is not None and trial.status != "dose_pair_safety_eliminated"
    ]
    expected_arm_gaps = []
    expected_factor_gaps = [[], []]
    for trial in pair_trials:
        arms = trial.arm_history
        factors = trial.factor_history
        arm1, arm2 = int(np.count_nonzero(arms == 1)), int(np.count_nonzero(arms == 2))
        expected_arm_gaps.append(abs(arm1 - arm2))
        if arm1 and arm2:
            expected_factor_gaps[0].append(
                abs(np.mean(factors[arms == 1, 0] == 1) - np.mean(factors[arms == 2, 0] == 1))
            )
            expected_factor_gaps[1].append(
                abs(np.mean(factors[arms == 1, 1] == 1) - np.mean(factors[arms == 2, 1] == 1))
            )
    assert summary.mean_pair_arm_count_imbalance.mean == pytest.approx(
        float(np.mean(expected_arm_gaps))
    )
    for observed, expected in zip(
        summary.mean_factor_level1_proportion_imbalance, expected_factor_gaps, strict=True
    ):
        assert observed.mean == pytest.approx(float(np.mean(expected)))


def test_prior_all_overdose_trials_remain_in_no_selection_denominators():
    design, toxicity, response, stage_two = _inputs(prior=BARDLogisticPrior([2.0, 0.0], [0.0, 0.0]))
    summary = simulate_bard_blrm(
        design,
        toxicity,
        response,
        stage_two,
        true_obd_noninferiority=2,
        true_obd_utility=2,
        trials=2,
        rng=19,
    )
    assert summary.no_pair_trials == 2
    assert summary.noninferiority_no_selection.count == 2
    assert summary.utility_no_selection.count == 2
    assert summary.noninferiority_accuracy.correct_all_trials.probability == 0.0
    assert summary.utility_accuracy.correct_all_trials.probability == 0.0
    assert summary.mean_pair_arm_count_imbalance.count == 0
    assert all(item.count == 0 for item in summary.mean_factor_level1_proportion_imbalance)


def test_aggregate_minimum_work_failure_precedes_generator_consumption():
    design, toxicity, response, stage_two = _inputs()
    generator = np.random.default_rng(2026)
    before = copy.deepcopy(generator.bit_generator.state)
    minimum_per_trial = design.chains * design.draws * len(design.doses)
    with pytest.raises(ValueError, match="one minimum BF-BLRM fit per trial"):
        simulate_bard_blrm(
            design,
            toxicity,
            response,
            stage_two,
            true_obd_noninferiority=2,
            true_obd_utility=2,
            trials=2,
            rng=generator,
            max_total_work=2 * minimum_per_trial - 1,
        )
    assert generator.bit_generator.state == before
