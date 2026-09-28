import copy

import numpy as np
import pytest
from scipy.special import gammaincinv

from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design
from mdanderson_stats.bop2_dc_survival_optimization import optimize_bop2_dc_survival


def _one_patient_exact_probabilities(design, true_median, followup):
    shape = design.prior_shape + 1
    mean = true_median / np.log(2.0)
    lower_boundary = (
        design.lrv * gammaincinv(shape, design.lambda_lrv) / np.log(2.0) - design.prior_scale
    )
    clinical_boundary = (
        design.cmv * gammaincinv(shape, design.lambda_cmv) / np.log(2.0) - design.prior_scale
    )
    event_probability = -np.expm1(-followup / mean)
    go_boundary = max(lower_boundary, clinical_boundary, 0.0)
    no_go_boundary = min(lower_boundary, clinical_boundary, followup)
    go_event = (
        np.exp(-go_boundary / mean) - np.exp(-followup / mean) if go_boundary < followup else 0.0
    )
    no_go_event = 1.0 - np.exp(-no_go_boundary / mean) if no_go_boundary > 0 else 0.0
    consider_event = event_probability - go_event - no_go_event
    no_event_probability = np.exp(-followup / mean)
    no_event_decision = str(design.monitor(1, 0, followup).decision.item())
    probabilities = dict.fromkeys(("stop_no_go", "final_go", "final_consider", "final_no_go"), 0.0)
    probabilities["final_go"] += go_event
    probabilities["final_no_go"] += no_go_event
    probabilities["final_consider"] += consider_event
    probabilities[no_event_decision] += no_event_probability
    return np.asarray([probabilities[label] for label in design_labels])


design_labels = ("stop_no_go", "final_go", "final_consider", "final_no_go")


def test_one_patient_holdout_matches_exact_exponential_event_partition():
    followup = 3.0
    design = bop2_dc_survival_design(
        1,
        lrv=3.0,
        cmv=5.0,
        lambda_lrv=0.2,
        lambda_cmv=0.1,
        prior_shape=1.0,
        prior_scale=2.0,
        looks=[1],
    )
    result = optimize_bop2_dc_survival(
        1,
        3.0,
        5.0,
        3.0,
        8.0,
        lambda_lrv_grid=[0.2],
        lambda_cmv_grid=[0.1],
        gamma_lrv_grid=[0.5],
        gamma_cmv_grid=[0.5],
        prior_shape=1.0,
        prior_scale=2.0,
        looks=[1],
        accrual_rate=1.0,
        final_followup=followup,
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        n_trials=8192,
        n_validation=8192,
        rng=904,
    )

    assert result.candidate_count == result.selected_index + 1 == 1
    assert result.validation_feasible
    assert result.futile_decision_probability.shape == (1, 4)
    assert result.effective_decision_probability.shape == (1, 4)
    for truth, oc in (
        (3.0, result.validation_futile),
        (8.0, result.validation_effective),
    ):
        exact = _one_patient_exact_probabilities(design, truth, followup)
        allowance = 5 * np.sqrt(exact * (1 - exact) / oc.trials) + 2 / oc.trials
        np.testing.assert_array_less(np.abs(oc.decision_probability - exact), allowance)


def test_preflight_work_limit_does_not_advance_generator():
    generator = np.random.default_rng(16)
    state_before = copy.deepcopy(generator.bit_generator.state)

    with pytest.raises(ValueError, match="max_work"):
        optimize_bop2_dc_survival(
            4,
            1,
            2,
            2,
            4,
            lambda_lrv_grid=[0.5],
            lambda_cmv_grid=[0.2],
            gamma_lrv_grid=[0.0],
            gamma_cmv_grid=[0.0],
            prior_shape=1,
            prior_scale=1,
            looks=[2, 4],
            accrual_rate=1,
            final_followup=2,
            false_go_limit=0.1,
            false_no_go_limit=0.2,
            n_trials=10,
            n_validation=10,
            rng=generator,
            max_work=1,
        )
    assert generator.bit_generator.state == state_before

    with pytest.raises(ValueError, match="theta_effective >= cmv"):
        optimize_bop2_dc_survival(
            4,
            1,
            2,
            1,
            1.5,
            lambda_lrv_grid=[0.5],
            lambda_cmv_grid=[0.2],
            gamma_lrv_grid=[0.0],
            gamma_cmv_grid=[0.0],
            prior_shape=1,
            prior_scale=1,
            looks=[2, 4],
            accrual_rate=1,
            final_followup=2,
            false_go_limit=0.1,
            false_no_go_limit=0.2,
            n_trials=10,
            n_validation=10,
            rng=generator,
            max_work=10_000,
        )
    assert generator.bit_generator.state == state_before
