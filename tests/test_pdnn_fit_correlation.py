import numpy as np

from mdanderson_stats.pdnn_fit_correlation import pdnn_fit_correlations


def test_grouped_log_signal_pearson_correlations_and_undefined_groups():
    observed = np.array([1.0, 2.0, 4.0, 3.0, 4.0, 5.0, 5.0, 9.0])
    fitted = np.array([2.0, 4.0, 8.0, 8.0, 4.0, 2.0, 3.0, 7.0])
    ids = np.array([20, 20, 20, 5, 5, 9, 9, 12])
    result = pdnn_fit_correlations(observed, fitted, ids)

    np.testing.assert_array_equal(result.probeset_ids, [5, 9, 12, 20])
    np.testing.assert_allclose(result.correlation[[0, 3]], [-1.0, 1.0])
    assert np.isnan(result.correlation[1])  # constant observed log signal
    assert np.isnan(result.correlation[2])  # singleton
    assert not result.probeset_ids.flags.writeable
    assert not result.correlation.flags.writeable


def test_correlation_is_invariant_to_positive_intensity_rescaling():
    observed = np.array([1.0, 3.0, 8.0, 2.0, 5.0, 11.0])
    fitted = np.array([2.0, 4.0, 7.0, 3.0, 6.0, 10.0])
    ids = np.array([0, 0, 0, 1, 1, 1])
    baseline = pdnn_fit_correlations(observed, fitted, ids).correlation
    shifted = pdnn_fit_correlations(observed * 1e100, fitted * 1e-100, ids).correlation
    np.testing.assert_allclose(shifted, baseline, atol=2e-15)


def test_interleaved_constant_and_near_equal_large_intensity_groups():
    observed = np.array([3.0, 1e200, 3.0, 1e200 + 4e184, 3.0, 1e200 + 8e184])
    fitted = np.array([2.0, 5e200, 2.0, 5e200 + 8e184, 2.0, 5e200 + 16e184])
    ids = np.array([7, 2, 7, 2, 7, 2])
    result = pdnn_fit_correlations(observed, fitted, ids)
    assert np.isnan(result.correlation[1])  # exactly constant intensities
    assert result.correlation[0] > 0.99


def test_oversized_sequence_is_rejected_before_conversion():
    class Oversized:
        def __len__(self):
            return 500_001

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversized input must be rejected before conversion")

    import pytest

    with pytest.raises(ValueError, match="workspace limit"):
        pdnn_fit_correlations(Oversized(), [], [])
