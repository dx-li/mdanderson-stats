import numpy as np
import pytest

from mdanderson_stats.bop2_dc_normal_optimization import optimize_bop2_dc_normal


def _options():
    return dict(
        max_subjects=8,
        theta_lrv=0.0,
        theta_cmv=0.5,
        theta_futile=-1.0,
        theta_effective=1.0,
        truth_sd=1.0,
        prior_mean=0.0,
        prior_precision=0.5,
        prior_shape=1.5,
        prior_scale=0.75,
        looks=[4, 8],
        n_trials=48,
        n_validation=40,
        lambda_lrv_grid=[0.6, 0.8],
        lambda_cmv_grid=[0.3, 0.5],
        gamma_lrv_grid=[0.0],
        gamma_cmv_grid=[0.0, 0.5],
    )


def test_finite_grid_calibration_has_fresh_validation_and_replay_seed() -> None:
    result = optimize_bop2_dc_normal(**_options(), rng=341)
    replay = optimize_bop2_dc_normal(**_options(), rng=result.rng_seed)
    np.testing.assert_array_equal(
        result.validation_oc.decision_probability, replay.validation_oc.decision_probability
    )
    np.testing.assert_array_equal(
        result.calibration_oc.decision_probability, replay.calibration_oc.decision_probability
    )
    assert result.candidates.parameters.shape == (8, 4)
    assert result.selected_index >= 0
    assert result.candidates.decision_probability.shape == (2, 8, 4)
    assert result.candidates.decision_mcse.shape == (2, 8, 4)
    assert result.candidates.expected_sample_size.shape == (2, 8)
    assert result.candidates.enrollment_mcse.shape == (2, 8)
    assert result.calibration_oc.n_trials == 48
    assert result.validation_oc.n_trials == 40
    assert result.validation_oc.rng_seed != result.calibration_oc.rng_seed
    assert np.allclose(result.calibration_oc.decision_probability.sum(axis=1), 1)
    assert np.allclose(result.validation_oc.decision_probability.sum(axis=1), 1)
    assert np.all(result.calibration_oc.decision_mcse >= 0)
    assert result.validation_feasible == (
        result.validation_oc.decision_probability[0, 1] <= result.false_go_limit
        and result.validation_oc.decision_probability[1, 0]
        + result.validation_oc.decision_probability[1, 3]
        <= result.false_no_go_limit
    )


def test_constraints_and_preflight_fail_before_rng_consumption() -> None:
    options = _options()
    options.update(n_trials=100_000, n_validation=100_000)
    rng1, rng2 = np.random.default_rng(10), np.random.default_rng(10)
    with pytest.raises(ValueError, match="one-million-cell"):
        optimize_bop2_dc_normal(**options, rng=rng1)
    assert rng1.random() == rng2.random()

    options = _options()
    options.update(theta_effective=0.4)
    with pytest.raises(ValueError, match="theta_effective >= theta_cmv"):
        optimize_bop2_dc_normal(**options, rng=rng1)
    assert rng1.random() == rng2.random()


def test_common_path_tails_preserve_large_affine_offset() -> None:
    from mdanderson_stats.bop2_dc_normal import bop2_dc_normal_design
    from mdanderson_stats.bop2_dc_normal_optimization import _tail_paths

    options = dict(
        max_subjects=8,
        theta_lrv=0.0,
        theta_cmv=0.5,
        prior_mean=0.0,
        prior_precision=0.5,
        prior_shape=1.5,
        prior_scale=0.2,
        looks=[4, 8],
    )
    base = bop2_dc_normal_design(**options)
    shift = 1e15
    shifted = bop2_dc_normal_design(
        **{**options, "theta_lrv": shift, "theta_cmv": shift + 0.5, "prior_mean": shift}
    )
    standardized = np.array(
        [
            [
                -2.123456789,
                -0.234567891,
                0.123456789,
                1.345678912,
                -1.012345678,
                0.876543219,
                1.987654321,
                2.012345678,
            ],
            [
                2.987654321,
                1.234567891,
                0.345678912,
                -1.765432109,
                1.456789123,
                0.234567891,
                -0.876543219,
                -2.123456789,
            ],
            [
                -1.123456789,
                -0.876543219,
                1.012345678,
                1.234567891,
                -2.987654321,
                -1.876543219,
                2.345678912,
                2.876543219,
            ],
            [
                1.345678912,
                0.123456789,
                -1.012345678,
                -0.234567891,
                2.123456789,
                1.876543219,
                -2.234567891,
                -1.765432109,
            ],
        ],
        dtype=float,
    )
    base_tails = _tail_paths(standardized, 0.0, 1.0, base.looks, base)
    shifted_tails = _tail_paths(standardized, shift, 1.0, shifted.looks, shifted)
    np.testing.assert_allclose(shifted_tails[0], base_tails[0], rtol=0, atol=2e-15)
    np.testing.assert_allclose(shifted_tails[1], base_tails[1], rtol=0, atol=2e-15)
