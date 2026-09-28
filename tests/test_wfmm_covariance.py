import numpy as np
import pytest

from mdanderson_stats.wfmm_basis import wfmm_basis
from mdanderson_stats.wfmm_covariance import wfmm_covariance, wfmm_summarize_covariance


def test_haar_equal_variances_reconstruct_identity_covariance():
    basis = wfmm_basis(4, transform="wavelet", filter_length=2, levels=1)
    covariance = wfmm_covariance(np.full(4, 3.0), basis)
    np.testing.assert_allclose(covariance, 3.0 * np.eye(4), atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_custom_nonsymmetric_basis_orientation_and_posterior_summaries():
    angle = 0.37
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    basis = wfmm_basis(2, transform="custom", custom_matrix=rotation)
    draws = np.asarray([[[[4.0, 1.0]], [[1.0, 3.0]], [[2.0, 2.0]]]])
    summary = wfmm_summarize_covariance(
        draws,
        basis,
        quantiles=[0.25, 0.5, 0.75],
        include_covariance=True,
        retain_covariance_draws=True,
    )

    expected_mean_omega = np.asarray([7 / 3, 2.0])
    expected_covariance = rotation @ np.diag(expected_mean_omega) @ rotation.T
    expected_first_covariance = rotation @ np.diag([4.0, 1.0]) @ rotation.T
    expected_function_draws = np.asarray(
        [
            np.diag(rotation @ np.diag(row) @ rotation.T)
            for row in [[4.0, 1.0], [1.0, 3.0], [2.0, 2.0]]
        ]
    )
    np.testing.assert_allclose(summary.variance_mean, expected_mean_omega[None, :])
    np.testing.assert_allclose(summary.covariance_mean[0], expected_covariance, atol=1e-14)
    np.testing.assert_allclose(
        summary.variance_function_mean[0], expected_function_draws.mean(axis=0)
    )
    np.testing.assert_allclose(
        summary.variance_function_standard_deviation[0], expected_function_draws.std(axis=0, ddof=1)
    )
    np.testing.assert_allclose(
        summary.variance_function_quantiles[:, 0],
        np.quantile(expected_function_draws, [0.25, 0.5, 0.75], axis=0, method="linear"),
    )
    expected_plugin_correlation = expected_covariance / np.sqrt(
        np.diag(expected_covariance)[:, None] * np.diag(expected_covariance)[None, :]
    )
    np.testing.assert_allclose(
        summary.correlation_from_mean_variance[0], expected_plugin_correlation
    )
    np.testing.assert_allclose(summary.covariance_draws[0, 0, 0], expected_first_covariance)
    assert not summary.covariance_mean.flags.writeable
    assert not summary.correlation_from_mean_variance.flags.writeable


def test_posterior_summary_bounds_full_covariance_draw_product_before_expansion():
    basis = wfmm_basis(32, transform="identity")
    draws = np.zeros((1, 3000, 1, 32))
    with pytest.raises(ValueError, match="outputs exceed 2000000 cells"):
        wfmm_summarize_covariance(
            draws,
            basis,
            include_covariance=True,
            retain_covariance_draws=True,
        )
