import numpy as np
import pytest

from mdanderson_stats.wfmm_empirical_bayes import calibrate_wfmm_shrinkage
from mdanderson_stats.wfmm_model import WFMMPrior, fit_wfmm_coefficients


def test_gls_conditional_variances_and_partition_specific_em_fixed_points():
    design = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    beta = np.asarray([[1.0, 3.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
    values = design @ beta
    values[:, 2:] = 0.0
    result = calibrate_wfmm_shrinkage(
        values,
        design,
        random_variance=None,
        residual_variance=1.0,
        coefficient_partition=[0, 0, 1, 1],
        max_iterations=2000,
    )

    np.testing.assert_allclose(result.fixed_estimates, beta)
    information = design.T @ design
    np.testing.assert_allclose(result.conditional_variances[:, 0], 1 / np.diag(information))
    assert not np.isclose(result.conditional_variances[0, 0], np.linalg.inv(information)[0, 0])
    np.testing.assert_array_equal(result.partition_labels, [0, 1])
    np.testing.assert_array_equal(result.group_slab_to_sampling_variance[:, 1], [0.0, 0.0])
    np.testing.assert_array_equal(result.group_inclusion_probability[:, 1], [0.0, 0.0])

    z = result.standardized_estimates[0, :2]
    pi = result.group_inclusion_probability[0, 0]
    u = result.group_slab_to_sampling_variance[0, 0]
    if u > 0.0:
        posterior_inclusion = 1 / (
            1 + np.exp(-(np.log(pi) - np.log1p(-pi) - 0.5 * np.log1p(u) + 0.5 * z**2 * u / (1 + u)))
        )
        expected_u = max(0.0, np.dot(posterior_inclusion, z**2) / posterior_inclusion.sum() - 1)
        np.testing.assert_allclose(pi, posterior_inclusion.mean(), rtol=1e-7)
        np.testing.assert_allclose(u, expected_u, rtol=1e-7)

    np.testing.assert_array_equal(result.prior.inclusion_probability[:, 2:], 0.0)
    np.testing.assert_array_equal(result.prior.slab_variance[:, 2:], 0.0)
    assert not result.fixed_estimates.flags.writeable


def test_all_spike_prior_is_valid_for_the_existing_coefficient_sampler():
    prior = WFMMPrior(inclusion_probability=0.0, slab_variance=0.0)
    fitted = fit_wfmm_coefficients(
        np.zeros((4, 1)),
        np.ones((4, 1)),
        prior=prior,
        residual_variance=1.0,
        estimate_variances=False,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(14),
    )
    np.testing.assert_array_equal(fitted.coefficients, 0.0)
    np.testing.assert_array_equal(fitted.inclusion_indicators, False)

    with pytest.raises(ValueError, match="only when inclusion_probability is zero"):
        WFMMPrior(inclusion_probability=0.5, slab_variance=0.0)
    with pytest.raises(ValueError, match="array shapes must match"):
        WFMMPrior(inclusion_probability=np.zeros((1, 3)), slab_variance=np.zeros((3, 1)))


def test_calibration_rejects_rank_deficiency_and_reports_iteration_limit():
    with pytest.raises(ArithmeticError, match="not full rank"):
        calibrate_wfmm_shrinkage(
            np.ones((4, 1)),
            np.ones((4, 2)),
            random_variance=None,
            residual_variance=1.0,
        )

    nearly_collinear = np.column_stack(
        (np.ones(4), np.ones(4) + np.asarray([0.0, 1e-8, 2e-8, 3e-8]))
    )
    with pytest.raises(ArithmeticError, match="numerically ill-conditioned"):
        calibrate_wfmm_shrinkage(
            np.ones((4, 1)),
            nearly_collinear,
            random_variance=None,
            residual_variance=1.0,
        )

    x = np.ones((2, 1))
    y = np.asarray([[3.0, -3.0], [3.0, -3.0]])
    result = calibrate_wfmm_shrinkage(
        y,
        x,
        random_variance=None,
        residual_variance=1.0,
        coefficient_partition=[0, 0],
        max_iterations=1,
    )
    np.testing.assert_array_equal(result.group_iterations, [[1]])
    np.testing.assert_array_equal(result.group_converged, [[False]])
