import numpy as np
import pytest
from scipy.special import gammainc

from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design


def test_inverse_gamma_median_posterior_uses_observed_time():
    design = bop2_dc_survival_design(4, 1.0, 2.0, prior_shape=2.0, prior_scale=3.0, looks=[2, 4])
    state = design.monitor([2, 4], [0, 1], [1.5, 2.0])
    expected_shape = np.array([2.0, 3.0])
    expected_scale = np.array([4.5, 5.0])
    np.testing.assert_allclose(state.posterior_shape, expected_shape)
    np.testing.assert_allclose(state.posterior_scale, expected_scale)
    np.testing.assert_allclose(
        state.posterior_lrv, gammainc(expected_shape, expected_scale * np.log(2.0))
    )
    assert state.posterior_median_scale.flags.writeable is False


def test_zero_time_no_event_is_valid_and_broadcasts():
    design = bop2_dc_survival_design(2, 1.0, 2.0, prior_shape=2.0, prior_scale=3.0, looks=[2])
    state = design.monitor([0, 2], 0, 0)
    assert state.sample_size.tolist() == [0, 2]
    assert state.events.tolist() == [0, 0]
    assert np.all(np.isfinite(state.posterior_lrv))


def test_invalid_time_and_event_summaries_are_rejected():
    design = bop2_dc_survival_design(2, 1.0, 2.0, looks=[2])
    with pytest.raises(ValueError, match="events"):
        design.monitor(1, 2, 1.0)
    with pytest.raises(ValueError, match="total_time"):
        design.monitor(1, 0, -1.0)


def test_tiny_prior_remains_finite_at_zero_exposure():
    design = bop2_dc_survival_design(
        1, 1e-300, 2e-300, prior_shape=1e-300, prior_scale=1e-300, looks=[1]
    )
    state = design.monitor(0, 0, 0.0)
    assert np.all(np.isfinite(state.posterior_lrv))


def test_time_unit_rescaling_preserves_posterior_tails():
    base = bop2_dc_survival_design(4, 1.0, 2.0, prior_shape=2.0, prior_scale=3.0, looks=[4])
    scaled = bop2_dc_survival_design(4, 1e150, 2e150, prior_shape=2.0, prior_scale=3e150, looks=[4])
    small = bop2_dc_survival_design(
        4, 1e-150, 2e-150, prior_shape=2.0, prior_scale=3e-150, looks=[4]
    )
    first = base.monitor(2, 1, 1.5)
    second = scaled.monitor(2, 1, 1.5e150)
    third = small.monitor(2, 1, 1.5e-150)
    np.testing.assert_allclose(first.posterior_lrv, second.posterior_lrv)
    np.testing.assert_allclose(first.posterior_cmv, second.posterior_cmv)
    np.testing.assert_allclose(first.posterior_lrv, third.posterior_lrv)
    np.testing.assert_allclose(first.posterior_cmv, third.posterior_cmv)


def test_overflowed_raw_scale_keeps_representable_median_scale():
    design = bop2_dc_survival_design(
        2, 1e308, 1.5e308, prior_shape=2.0, prior_scale=1e308, looks=[2]
    )
    state = design.monitor(1, 0, 1e308)
    assert np.isinf(state.posterior_scale)
    np.testing.assert_allclose(state.posterior_median_scale, (1e308 * np.log(2.0)) * 2)
    assert np.all(np.isfinite(state.posterior_lrv))
