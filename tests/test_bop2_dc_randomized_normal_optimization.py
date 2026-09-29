import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design
from mdanderson_stats.bop2_dc_randomized_normal_optimization import (
    optimize_bop2_dc_randomized_normal,
)


def _run(offset: float = 0.0):
    design = bop2_dc_randomized_normal_design(
        4,
        -0.25,
        0.25,
        control_prior=(offset, 1.0, 2.0, 1.0),
        treatment_prior=(offset, 1.0, 2.0, 1.0),
        arm_assignments=(0, 1, 0, 1),
        looks=(2, 4),
        lambda_lrv=0.5,
        lambda_cmv=0.5,
        gamma_lrv=0.5,
        gamma_cmv=0.5,
        comparison_tolerance=1e-6,
        quadrature_limit=40,
    )
    return optimize_bop2_dc_randomized_normal(
        design,
        (offset, offset),
        (offset, offset + 0.5),
        futile_truth_sd=(0.5, 0.5),
        effective_truth_sd=(0.5, 0.5),
        lambda_lrv_grid=(0.4, 0.6),
        lambda_cmv_grid=(0.4, 0.6),
        gamma_lrv_grid=(0.5,),
        gamma_cmv_grid=(0.5,),
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        n_trials=4,
        n_validation=3,
        rng=913,
    )


def test_randomized_normal_calibration_retains_grid_and_independent_holdout() -> None:
    result = _run()
    assert result.candidates.parameters.shape == (4, 4)
    assert result.candidates.decision_probability.shape == (2, 4, 5)
    np.testing.assert_allclose(result.candidates.decision_probability.sum(axis=2), 1.0)
    assert result.calibration_oc.rng_seed != result.validation_oc.rng_seed
    assert result.calibration_oc.decision_probability.shape == (2, 5)
    assert result.validation_oc.decision_probability.shape == (2, 5)
    assert not result.candidates.parameters.flags.writeable


def test_common_location_shift_preserves_normal_calibration_paths() -> None:
    baseline = _run()
    shifted = _run(2.0**40)
    assert baseline.selected_index == shifted.selected_index
    np.testing.assert_array_equal(
        baseline.candidates.decision_probability,
        shifted.candidates.decision_probability,
    )
    np.testing.assert_array_equal(
        baseline.validation_oc.expected_sample_size,
        shifted.validation_oc.expected_sample_size,
    )


def test_candidate_cutoff_underflow_is_rejected_before_simulation() -> None:
    design = bop2_dc_randomized_normal_design(
        4,
        -0.25,
        0.25,
        control_prior=(0.0, 1.0, 2.0, 1.0),
        treatment_prior=(0.0, 1.0, 2.0, 1.0),
        arm_assignments=(0, 1, 0, 1),
        looks=(2, 4),
        comparison_tolerance=1e-6,
        quadrature_limit=40,
    )
    with pytest.raises(ArithmeticError, match="cutoff underflows"):
        optimize_bop2_dc_randomized_normal(
            design,
            (0.0, 0.0),
            (0.0, 0.5),
            futile_truth_sd=(0.5, 0.5),
            effective_truth_sd=(0.5, 0.5),
            lambda_lrv_grid=(0.5, 5e-324),
            lambda_cmv_grid=(0.5,),
            gamma_lrv_grid=(1.0,),
            gamma_cmv_grid=(0.5,),
            n_trials=2,
            n_validation=2,
            rng=1,
        )
