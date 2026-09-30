import numpy as np
import pytest

from mdanderson_stats.wfmm_basis import wfmm_basis, wfmm_inverse, wfmm_transform
from mdanderson_stats.wfmm_covariance import wfmm_covariance, wfmm_summarize_covariance


def _nonorthogonal_pair(scale=1.0):
    analysis = scale * np.asarray([[2.0, 1.0], [0.0, 1.0]])
    synthesis = np.asarray([[0.5, -0.5], [0.0, 1.0]]) / scale
    return analysis, synthesis


def test_paired_custom_transform_reconstructs_and_propagates_covariance():
    analysis, synthesis = _nonorthogonal_pair()
    basis = wfmm_basis(
        2,
        transform="custom",
        analysis_matrix=analysis,
        synthesis_matrix=synthesis,
    )
    curves = np.asarray([[1.25, -2.0], [3.0, 0.75]])
    transformed = wfmm_transform(curves, basis)
    np.testing.assert_allclose(transformed.coefficients, curves @ analysis, atol=1e-14)
    np.testing.assert_allclose(wfmm_inverse(transformed.coefficients, basis), curves, atol=1e-14)

    omega = np.asarray([4.0, 9.0])
    expected_covariance = synthesis.T @ np.diag(omega) @ synthesis
    np.testing.assert_allclose(wfmm_covariance(omega, basis), expected_covariance)
    draws = omega.reshape(1, 1, 1, 2)
    summary = wfmm_summarize_covariance(
        np.concatenate((draws, 2 * draws), axis=1), basis, include_covariance=True
    )
    np.testing.assert_allclose(
        summary.variance_function_mean[0], np.diag(1.5 * expected_covariance)
    )
    np.testing.assert_allclose(summary.covariance_mean[0], 1.5 * expected_covariance)


@pytest.mark.parametrize("scale,variance", [(1e-200, 1e-200), (1e200, 1e200)])
def test_paired_custom_covariance_rescaling_avoids_premature_overflow(scale, variance):
    analysis, synthesis = _nonorthogonal_pair(scale)
    basis = wfmm_basis(
        2,
        transform="custom",
        analysis_matrix=analysis,
        synthesis_matrix=synthesis,
    )
    covariance = wfmm_covariance([variance, variance], basis)
    base_synthesis = _nonorthogonal_pair()[1]
    factor = (np.sqrt(variance) / scale) ** 2
    np.testing.assert_allclose(covariance, factor * (base_synthesis.T @ base_synthesis))


@pytest.mark.parametrize(
    "synthesis_scale,variance,expected",
    [(1e200, 1e-200, 1e200), (1e-200, 1e300, 1e-100)],
)
def test_square_inverse_extreme_scale_covariance_and_variance_function(
    synthesis_scale, variance, expected
):
    identity = np.eye(2)
    basis = wfmm_basis(
        2,
        transform="custom",
        analysis_matrix=identity / synthesis_scale,
        synthesis_matrix=identity * synthesis_scale,
    )
    covariance = wfmm_covariance([variance, variance], basis)
    np.testing.assert_allclose(covariance, expected * identity, rtol=2e-15)
    draws = np.full((1, 2, 1, 2), variance)
    summary = wfmm_summarize_covariance(draws, basis, include_covariance=True)
    np.testing.assert_allclose(summary.variance_function_mean[0], expected * np.ones(2))
    np.testing.assert_allclose(summary.covariance_mean[0], expected * identity, rtol=2e-15)


def test_pair_validation_is_explicit_and_existing_orthogonal_api_remains():
    identity = np.eye(2)
    with pytest.raises(ValueError, match="supplied together"):
        wfmm_basis(2, transform="custom", analysis_matrix=identity)
    with pytest.raises(ValueError, match="numerical inverses"):
        wfmm_basis(2, transform="custom", analysis_matrix=identity, synthesis_matrix=2 * identity)
    ill_conditioned = np.asarray([[1.0, 1e16], [0.0, 1.0]])
    ill_conditioned_inverse = np.asarray([[1.0, -1e16], [0.0, 1.0]])
    with pytest.raises(ValueError, match="condition-number limit"):
        wfmm_basis(
            2,
            transform="custom",
            analysis_matrix=ill_conditioned,
            synthesis_matrix=ill_conditioned_inverse,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        wfmm_basis(
            2,
            transform="custom",
            custom_matrix=identity,
            analysis_matrix=identity,
            synthesis_matrix=identity,
        )
    with pytest.raises(ValueError, match="square matrices"):
        wfmm_basis(
            2,
            transform="custom",
            analysis_matrix=np.ones((2, 3)),
            synthesis_matrix=np.ones((3, 2)),
        )

    old_api = wfmm_basis(2, transform="custom", custom_matrix=identity)
    np.testing.assert_array_equal(wfmm_transform([[1.0, 2.0]], old_api).coefficients, [[1.0, 2.0]])
    np.testing.assert_array_equal(wfmm_inverse([[1.0, 2.0]], old_api), [[1.0, 2.0]])
