import numpy as np

from mdanderson_stats.wfmm_model import WFMMPrior, fit_wfmm_coefficients


def test_fixed_variance_wavelet_coefficient_posterior_matches_exact_mixture() -> None:
    fixed = np.column_stack((np.ones(5), [-1.0, -0.5, 0.0, 0.5, 1.0]))
    random = np.eye(3)[[0, 0, 1, 1, 2]]
    observed = np.array([[0.2, -0.4], [0.6, 0.1], [-0.1, 0.7], [0.5, 0.3], [1.1, -0.2]])
    prior = WFMMPrior(
        np.tile(np.array([0.7, 0.4])[:, None], (1, 2)),
        np.tile(np.array([0.8, 0.5])[:, None], (1, 2)),
    )
    fit = fit_wfmm_coefficients(
        observed,
        fixed,
        random,
        prior=prior,
        random_variance=[0.3, 0.1],
        residual_variance=[0.2, 0.4],
        estimate_variances=False,
        sample_random_effects=True,
        draws=4000,
        warmup=300,
        chains=2,
        rng=np.random.default_rng(17),
    )

    expected_mean = np.array([[0.2902688011, 0.0353251089], [0.1442001920, 0.0178030062]])
    expected_inclusion = np.array([[0.6653822649, 0.4603761069], [0.3744874106, 0.2698117499]])
    np.testing.assert_allclose(fit.summary.mean, expected_mean, atol=0.015)
    np.testing.assert_allclose(
        fit.inclusion_indicators.mean(axis=(0, 1)), expected_inclusion, atol=0.025
    )
    assert fit.random_effects is not None
    assert fit.random_effects.shape == (2, 4000, 3, 2)
    assert np.all(np.isfinite(fit.random_effects))
    assert not fit.coefficients.flags.writeable


def test_residual_variance_mh_matches_conjugate_precision_mean() -> None:
    observed = np.array([0.2, -0.4, 0.1, 0.7, -0.3])[:, None]
    prior = WFMMPrior(
        0.0,
        1.0,
        residual_shape=4.0,
        residual_scale=2.0,
    )
    fit = fit_wfmm_coefficients(
        observed,
        np.zeros((5, 1)),
        prior=prior,
        residual_variance=1.0,
        estimate_variances=True,
        proposal_sd=(np.empty((0, 1)), 0.2),
        draws=3000,
        warmup=500,
        chains=2,
        rng=np.random.default_rng(912),
    )

    posterior_shape = 4.0 + observed.shape[0] / 2
    posterior_scale = 2.0 + float(np.square(observed).sum()) / 2
    expected_mean_precision = posterior_shape / posterior_scale
    np.testing.assert_allclose(fit.coefficients, 0.0, atol=0.0)
    np.testing.assert_allclose(
        np.mean(1.0 / fit.residual_variances), expected_mean_precision, rtol=0.08
    )
    assert 0.0 < fit.variance_acceptance[0, 0] < 1.0


def test_uninformative_column_uses_proper_spike_and_slab_prior() -> None:
    fit = fit_wfmm_coefficients(
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        prior=WFMMPrior(1.0, 2.0),
        residual_variance=1.0,
        estimate_variances=False,
        draws=32,
        chains=2,
        rng=np.random.default_rng(81),
    )

    assert np.all(fit.inclusion_indicators)
    assert np.all(np.isfinite(fit.coefficients))
    assert np.any(fit.coefficients != 0.0)
