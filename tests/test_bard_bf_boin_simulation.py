import numpy as np

from mdanderson_stats.bard_bf_boin_simulation import simulate_bard_bf_boin
from mdanderson_stats.bard_bf_boin_trial import BARDStageTwoDesign, run_bard_bf_boin_trial
from mdanderson_stats.bard_response import BARDResponseModel, bard_response_model
from mdanderson_stats.bf_boin import BFBOINDesign


def _inputs(
    true_toxicity: list[float],
) -> tuple[BFBOINDesign, np.ndarray, BARDResponseModel, BARDStageTwoDesign]:
    design = BFBOINDesign(target=0.3, n_cap=6)
    toxicity = np.asarray(true_toxicity, dtype=float)
    response = bard_response_model([0.3, 0.6], [[1]], [1.0], [[1.0]])
    stage_two = BARDStageTwoDesign(
        total_target=6,
        eligible_profiles=[True],
        prior=[1.0, 1.0, 1.0, 1.0],
        safety_weights=[1.0, 1.0],
        toxicity_limit=0.3,
        efficacy_limit=0.3,
        safety_cutoff=0.95,
        efficacy_cutoff=0.95,
        utilities=[0.0, 30.0, 60.0, 100.0],
        margin=0.1,
        tie_arm=1,
        stage_two_accrual_rate=1.0,
    )
    return design, toxicity, response, stage_two


def test_streaming_oc_matches_serial_trial_replay() -> None:
    design, toxicity, response, stage_two = _inputs([0.1, 0.25])
    summary = simulate_bard_bf_boin(
        design,
        toxicity,
        response,
        stage_two,
        true_obd_noninferiority=2,
        true_obd_utility=1,
        trials=3,
        cohorts=2,
        cohort_size=3,
        rng=217,
    )

    generator = np.random.default_rng(217)
    reference = [
        run_bard_bf_boin_trial(
            design,
            toxicity,
            response,
            stage_two,
            cohorts=2,
            cohort_size=3,
            rng=generator,
        )
        for _ in range(3)
    ]
    sizes = np.asarray([item.total_sample_size for item in reference])
    durations = np.asarray([item.duration for item in reference])
    enrollment_mean = summary.mean_total_enrollment.mean
    duration_mean = summary.mean_duration.mean
    enrollment_sd = summary.mean_total_enrollment.sample_sd
    assert enrollment_mean is not None and duration_mean is not None and enrollment_sd is not None
    np.testing.assert_allclose(enrollment_mean, sizes.mean())
    np.testing.assert_allclose(duration_mean, durations.mean())
    np.testing.assert_allclose(
        enrollment_sd,
        sizes.std(ddof=1),
    )
    assert summary.trials == 3
    assert sum(item.count for item in summary.status_frequency) == 3
    assert summary.noninferiority_accuracy.correct_all_trials.denominator == 3
    assert summary.utility_accuracy.correct_all_trials.denominator == 3
    assert (
        summary.no_pair_trials
        + summary.safety_rejected_pair_trials
        + summary.arm_count_metric_trials
        == 3
    )
    assert (
        summary.factor_metric_trials + summary.empty_arm_factor_metric_trials
        == summary.arm_count_metric_trials
    )
    assert all(
        item.count == summary.factor_metric_trials
        for item in summary.mean_factor_level1_proportion_imbalance
    )
    # Only compact summaries are retained; the returned object has no trial ledger.
    assert not hasattr(summary, "trial_results")


def test_no_mtd_trials_have_explicit_no_pair_denominators() -> None:
    design, toxicity, response, stage_two = _inputs([1.0, 1.0])
    summary = simulate_bard_bf_boin(
        design,
        toxicity,
        response,
        stage_two,
        true_obd_noninferiority=1,
        true_obd_utility=2,
        trials=2,
        cohorts=1,
        cohort_size=3,
        rng=11,
    )

    assert summary.no_pair_trials == 2
    assert summary.arm_count_metric_trials == 0
    assert summary.factor_metric_trials == 0
    assert summary.mean_pair_arm_count_imbalance.count == 0
    assert summary.mean_pair_arm_count_imbalance.mean is None
    assert all(
        item.count == 0 and item.mean is None
        for item in summary.mean_factor_level1_proportion_imbalance
    )
    assert summary.noninferiority_no_selection.count == 2
    assert summary.utility_no_selection.count == 2
