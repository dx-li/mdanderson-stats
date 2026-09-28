import numpy as np
import pytest

from mdanderson_stats.bacis_ess import bacis_equivalent_sample_size


def _draws_with_sample_variance(variance, groups=1):
    delta = np.sqrt(np.asarray(variance, dtype=float) / 2.0)
    center = np.full(delta.shape, 0.5)
    return np.stack((center - delta, center + delta), axis=0)[:, :, None]


def test_fixed_response_count_variance_match_includes_zero_response_group():
    variance = 26.0 / (27.0**2 * 28.0)
    draws = _draws_with_sample_variance([variance])
    result = bacis_equivalent_sample_size(draws, [0], [25])
    np.testing.assert_allclose(result.posterior_variance, [variance], rtol=2e-14)
    np.testing.assert_allclose(result.equivalent_sample_size, [25.0], atol=2e-9)
    np.testing.assert_allclose(result.achieved_variance, result.posterior_variance, rtol=2e-12)
    assert result.candidate_roots[0].size == 1
    assert result.variance_residual[0] < 1e-14
    assert not result.equivalent_sample_size.flags.writeable
    boundary = bacis_equivalent_sample_size(_draws_with_sample_variance([1 / 12]), [0], [25])
    assert boundary.equivalent_sample_size[0] == pytest.approx(0.0, abs=1e-12)


def test_multiple_variance_roots_use_native_observed_rate_distance():
    variance = 33.0 / (14.0**2 * 15.0)
    result = bacis_equivalent_sample_size(_draws_with_sample_variance([variance]), [10], [25])
    np.testing.assert_allclose(result.candidate_roots[0], [12.0, 19.244564703609665], rtol=2e-10)
    assert result.equivalent_sample_size[0] == pytest.approx(19.244564703609665, rel=2e-10)
    assert result.candidate_rate_distance[0][1] < result.candidate_rate_distance[0][0]


def test_missing_or_nonfinite_equivalent_sizes_fail_instead_of_clamping():
    with pytest.raises(ValueError, match="no beta-equivalent"):
        bacis_equivalent_sample_size(np.array([[[0.0], [1.0]]]), [0], [25])
    with pytest.raises(ValueError, match="variance must be positive"):
        bacis_equivalent_sample_size(np.array([[[0.2], [0.2]]]), [1], [25])
    with pytest.raises(ValueError, match="total samples"):
        bacis_equivalent_sample_size(np.array([[[0.2]]]), [1], [25])
