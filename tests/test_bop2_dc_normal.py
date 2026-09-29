import numpy as np
import pytest
from scipy.stats import t as student_t

from mdanderson_stats.bop2_dc_normal import (
    bop2_dc_normal_design,
    run_bop2_dc_normal_trial,
    simulate_bop2_dc_normal,
)


def _design(**overrides):
    options = dict(
        max_subjects=8,
        theta_lrv=0.0,
        theta_cmv=0.5,
        prior_mean=0.0,
        prior_precision=0.5,
        prior_shape=1.5,
        prior_scale=0.75,
        lambda_lrv=0.8,
        lambda_cmv=0.5,
        gamma_lrv=0.5,
        gamma_cmv=0.5,
        looks=[4, 8],
    )
    options.update(overrides)
    return bop2_dc_normal_design(**options)


def test_nig_posterior_tails_match_complete_normal_sufficient_statistics() -> None:
    values = np.array([-1.0, 0.5, 2.0, 3.0])
    design = _design()
    state = design.monitor(values)
    n = values.size
    mean = float(np.mean(values))
    ss = float(np.sum((values - mean) ** 2))
    k = design.prior_precision + n
    posterior_location = (design.prior_precision * design.prior_mean + n * mean) / k
    posterior_shape = design.prior_shape + n / 2
    posterior_scale = (
        design.prior_scale
        + ss / 2
        + design.prior_precision * n * (mean - design.prior_mean) ** 2 / (2 * k)
    )
    posterior_df = 2 * posterior_shape
    scale = np.sqrt(posterior_scale / (posterior_shape * k))
    assert state.posterior_location == pytest.approx(posterior_location, rel=2e-14)
    assert state.posterior_location_centered == pytest.approx(
        posterior_location - values[0], rel=2e-14
    )
    assert state.posterior_df == pytest.approx(posterior_df)
    assert state.posterior_scale == pytest.approx(scale, rel=2e-14)
    assert state.posterior_lrv == pytest.approx(
        student_t.sf(design.theta_lrv, posterior_df, loc=posterior_location, scale=scale)
    )
    assert state.posterior_cmv == pytest.approx(
        student_t.sf(design.theta_cmv, posterior_df, loc=posterior_location, scale=scale)
    )


def test_trial_replay_stops_only_at_configured_interim_look() -> None:
    design = _design(
        max_subjects=6,
        theta_lrv=0.0,
        theta_cmv=1.0,
        prior_mean=0.0,
        prior_precision=1.0,
        prior_shape=2.0,
        prior_scale=0.5,
        lambda_lrv=0.99,
        lambda_cmv=0.99,
        gamma_lrv=0.0,
        gamma_cmv=0.0,
        looks=[2, 4, 6],
    )
    result = run_bop2_dc_normal_trial(design, [-8, -8, 1, 1, 1, 1])
    assert result.enrolled == 2
    assert result.decision == "stop_no_go"
    assert [state.sample_size for state in result.states] == [2]

    between_looks = design.monitor([-8, -8, 1])
    assert between_looks.decision == "continue"


def test_seeded_normal_operating_characteristics_are_replayable_and_bounded() -> None:
    design = _design(max_subjects=20, looks=[10, 20])
    first = simulate_bop2_dc_normal(design, 0.3, 1.1, n_trials=24, rng=913)
    replay = simulate_bop2_dc_normal(design, 0.3, 1.1, n_trials=24, rng=first.rng_seed)
    np.testing.assert_array_equal(first.sample_size, replay.sample_size)
    np.testing.assert_array_equal(first.decision, replay.decision)
    assert np.sum(first.decision_probability) == pytest.approx(1)
    np.testing.assert_allclose(
        first.decision_mcse,
        np.sqrt(first.decision_probability * (1 - first.decision_probability) / first.trials),
    )
    with pytest.raises(ValueError, match="patient-work"):
        simulate_bop2_dc_normal(design, 0.3, 1.1, n_trials=100_000, rng=913)


def test_monitor_is_invariant_to_large_affine_unit_offset() -> None:
    values = np.array([-0.5, 0.0, 0.25, 0.75])
    design = _design(
        max_subjects=8,
        theta_lrv=0.0,
        theta_cmv=0.5,
        prior_mean=0.0,
        prior_precision=0.5,
        prior_shape=1.5,
        prior_scale=0.2,
        looks=[4, 8],
    )
    shift = 1e15
    shifted_design = _design(
        max_subjects=8,
        theta_lrv=shift,
        theta_cmv=shift + 0.5,
        prior_mean=shift,
        prior_precision=0.5,
        prior_shape=1.5,
        prior_scale=0.2,
        looks=[4, 8],
    )
    base = design.monitor(values)
    shifted = shifted_design.monitor(values + shift)
    assert shifted.location_offset == shift + values[0]
    assert shifted.posterior_location_centered == pytest.approx(
        base.posterior_location_centered, abs=1e-15
    )
    assert shifted.posterior_lrv == pytest.approx(base.posterior_lrv, abs=2e-15)
    assert shifted.posterior_cmv == pytest.approx(base.posterior_cmv, abs=2e-15)
    assert shifted.decision == base.decision


def test_normal_simulation_is_affine_invariant_for_large_truth_offset() -> None:
    base_design = _design(max_subjects=20, looks=[10, 20])
    shift = 1e15
    shifted_design = _design(
        max_subjects=20,
        theta_lrv=shift,
        theta_cmv=shift + 0.5,
        prior_mean=shift,
        looks=[10, 20],
    )
    base = simulate_bop2_dc_normal(base_design, 0.0, 1.1, n_trials=32, rng=611)
    shifted = simulate_bop2_dc_normal(shifted_design, shift, 1.1, n_trials=32, rng=611)
    np.testing.assert_array_equal(shifted.sample_size, base.sample_size)
    np.testing.assert_array_equal(shifted.decision, base.decision)
    np.testing.assert_array_equal(shifted.decision_count, base.decision_count)
