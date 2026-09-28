import numpy as np
import pytest

from mdanderson_stats.bop2_dc_paired_optimization import (
    BOP2DCPairedInfeasibleError,
    optimize_bop2_dc_paired,
)


def _settings(**overrides):
    result = {
        "max_subjects": 4,
        "endpoint": "multiple_efficacy",
        "lrv": [0.2, 0.2],
        "cmv": [0.4, 0.5],
        "futile_probabilities": [0.1, 0.1, 0.1, 0.7],
        "effective_probabilities": [0.4, 0.2, 0.2, 0.2],
        "lambda_lrv_grid": [0.7],
        "lambda_cmv_grid": [0.3],
        "gamma_lrv_grid": [0.0],
        "gamma_cmv_grid": [0.0],
        "prior": [0.25, 0.25, 0.25, 0.25],
        "looks": [2, 4],
        "false_go_limit": 1.0,
        "false_no_go_limit": 1.0,
    }
    result.update(overrides)
    return result


def test_exact_grid_oc_matches_direct_joint_recursion_and_retains_association():
    result = optimize_bop2_dc_paired(**_settings())
    direct = result.design.operating_characteristics(
        np.array([result.futile_joint_probability, result.effective_joint_probability])
    )

    assert result.candidate_count == 1
    np.testing.assert_array_equal(result.feasible, [True])
    np.testing.assert_allclose(result.futile_joint_probability, [0.1, 0.1, 0.1, 0.7])
    np.testing.assert_allclose(result.effective_marginal_probability, [0.6, 0.6])
    np.testing.assert_allclose(result.futile_oc.final_go[0], direct.final_go[0])
    np.testing.assert_allclose(result.effective_oc.final_consider[0], direct.final_consider[1])
    np.testing.assert_allclose(result.false_go_rate[0], direct.final_go[0])
    np.testing.assert_allclose(result.false_no_go_rate[0], direct.no_go_probability[1])
    np.testing.assert_allclose(result.futile_oc.sample_size_probability.sum(axis=1), 1.0)


def test_asymmetric_endpoint_control_rows_and_ess_objective_are_preserved():
    result = optimize_bop2_dc_paired(
        **_settings(
            lambda_lrv_grid=[[0.7, 0.8], [0.8, 0.7]],
            lambda_cmv_grid=[[0.3, 0.2]],
            gamma_lrv_grid=[[0.0, 0.5]],
            gamma_cmv_grid=[[0.5, 0.0]],
            objective="ess_futile",
        )
    )

    assert result.candidate_count == 2
    assert result.objective == "ess_futile"
    assert result.grid["lambda_lrv"].shape == (2, 2)
    np.testing.assert_array_equal(
        result.design.lambda_lrv,
        result.grid["lambda_lrv"][result.selected_index],
    )
    expected_n = result.futile_oc.expected_sample_size
    assert expected_n[result.selected_index] == np.min(expected_n)


def test_mode_specific_effective_truth_and_exact_work_are_preflighted():
    with pytest.raises(ValueError, match="clinical-go rule"):
        optimize_bop2_dc_paired(**_settings(effective_probabilities=[0.1, 0.1, 0.1, 0.7]))

    with pytest.raises(ValueError, match="clinical-go rule"):
        optimize_bop2_dc_paired(
            **_settings(
                endpoint="efficacy_toxicity",
                lrv=[0.3, 0.2],
                cmv=[0.45, 0.15],
                effective_probabilities=[0.2, 0.3, 0.2, 0.3],
            )
        )

    with pytest.raises(ValueError, match="max_work"):
        optimize_bop2_dc_paired(**_settings(max_work=1))

    with pytest.raises(BOP2DCPairedInfeasibleError):
        optimize_bop2_dc_paired(**_settings(false_go_limit=0.0, false_no_go_limit=0.0))
