from copy import deepcopy

import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_survival import bop2_dc_randomized_survival_design
from mdanderson_stats.bop2_dc_randomized_survival_optimization import (
    optimize_bop2_dc_randomized_survival,
)
from mdanderson_stats.bop2_dc_randomized_survival_simulation import (
    simulate_bop2_dc_randomized_survival,
)


def _design(*, median_lrv=0.0, median_cmv=0.5, graduate=False):
    return bop2_dc_randomized_survival_design(
        max_subjects=4,
        median_lrv=median_lrv,
        median_cmv=median_cmv,
        control_prior=(2.0, 1.0),
        treatment_prior=(2.0, 1.0),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        graduate_at_interim=graduate,
        lambda_lrv=0.2,
        lambda_cmv=0.4,
    )


def _calibrate(design, *, rng=410):
    return optimize_bop2_dc_randomized_survival(
        design,
        (2.0, 3.0),
        (2.0, 5.0),
        lambda_lrv_grid=[0.2],
        lambda_cmv_grid=[0.4],
        gamma_lrv_grid=[0.5],
        gamma_cmv_grid=[0.5],
        accrual_rate=4.0,
        final_followup=0.5,
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        n_trials=4,
        n_validation=4,
        rng=rng,
    )


def test_finite_grid_reuses_replayable_holdout_and_conserves_decisions():
    result = _calibrate(_design())
    assert result.candidate_count == 1
    assert result.futile_decision_count.sum() == result.calibration_trials
    assert result.effective_decision_count.sum() == result.calibration_trials
    np.testing.assert_allclose(result.futile_decision_probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.effective_decision_probability.sum(axis=1), 1.0)
    replay = simulate_bop2_dc_randomized_survival(
        result.design,
        *result.futile_truth,
        accrual_rate=4.0,
        final_followup=0.5,
        n_trials=result.validation_trials,
        rng=result.validation_seed,
    )
    np.testing.assert_array_equal(replay.decision_count, result.validation_futile.decision_count)
    np.testing.assert_allclose(replay.mean_exposure, result.validation_futile.mean_exposure)
    assert result.work_units > 0
    assert not result.feasible.flags.writeable


def test_graduation_is_included_in_false_go_and_correct_go():
    design = bop2_dc_randomized_survival_design(
        max_subjects=4,
        median_lrv=0.0,
        median_cmv=0.5,
        control_prior=(2.0, 0.05),
        treatment_prior=(2.0, 100.0),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        graduate_at_interim=True,
        lambda_lrv=0.2,
        lambda_cmv=0.4,
    )
    result = _calibrate(design, rng=947)
    graduate = result.futile_decision_count[0, result.decision_labels.index("graduate")]
    assert graduate > 0
    assert result.false_go_rate[0] == graduate / result.calibration_trials
    effective_graduate = result.effective_decision_count[
        0, result.decision_labels.index("graduate")
    ]
    final_go = result.effective_decision_count[0, result.decision_labels.index("final_go")]
    assert result.correct_go_rate[0] == (effective_graduate + final_go) / result.calibration_trials


def test_futile_truth_is_not_restricted_to_lrv_and_preflight_keeps_rng_unchanged():
    # The caller-declared futile effect can exceed LRV while remaining below the
    # effective effect; this is intentionally not a validation error.
    design = _design(median_lrv=0.2, median_cmv=1.0)
    result = optimize_bop2_dc_randomized_survival(
        design,
        (1.0, 1.8),
        (1.0, 2.1),
        lambda_lrv_grid=[0.2],
        lambda_cmv_grid=[0.4],
        gamma_lrv_grid=[0.5],
        gamma_cmv_grid=[0.5],
        accrual_rate=4.0,
        final_followup=0.5,
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        n_trials=2,
        n_validation=2,
        rng=13,
    )
    assert result.futile_truth == (1.0, 1.8)

    large_design = bop2_dc_randomized_survival_design(
        max_subjects=1000,
        median_lrv=0.0,
        median_cmv=0.5,
        control_prior=(2.0, 1.0),
        treatment_prior=(2.0, 1.0),
        arm_assignments=[0, 1] * 500,
        looks=range(1, 1001),
        lambda_lrv=0.2,
        lambda_cmv=0.4,
    )
    rng = np.random.default_rng(9)
    state_before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="work budget"):
        optimize_bop2_dc_randomized_survival(
            large_design,
            (2.0, 2.5),
            (2.0, 4.0),
            lambda_lrv_grid=[0.2],
            lambda_cmv_grid=[0.4],
            gamma_lrv_grid=[0.5],
            gamma_cmv_grid=[0.5],
            accrual_rate=4.0,
            final_followup=0.5,
            false_go_limit=1.0,
            false_no_go_limit=1.0,
            n_trials=100,
            n_validation=100,
            rng=rng,
        )
    assert rng.bit_generator.state == state_before
