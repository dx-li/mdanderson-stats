import numpy as np
from scipy.special import gammaln
from scipy.stats import fisk, gamma, invgamma

from mdanderson_stats.tte_family_bayesian_gof import (
    _family_cdf,
    _family_log_likelihood,
    gamma_bayesian_gof,
    inverse_gamma_bayesian_gof,
    log_logistic_bayesian_gof,
)
from mdanderson_stats.weibull_unknown_shape_gof import _relative_log_times


def test_family_likelihoods_match_guide_densities_and_right_tails():
    times = np.array([0.7, 1.4, 3.2])
    event = np.array([True, False, True])
    relative, offset = _relative_log_times(times)
    shape, scale = 2.3, 1.7
    coordinates = np.array([np.log(shape), np.log(scale) - offset])
    for family, density, survival in (
        ("gamma", gamma.pdf(times, a=shape, scale=scale), gamma.sf(times, a=shape, scale=scale)),
        (
            "inverse_gamma",
            invgamma.pdf(times, a=shape, scale=scale),
            invgamma.sf(times, a=shape, scale=scale),
        ),
        (
            "log_logistic",
            fisk.pdf(times, c=shape, scale=scale),
            fisk.sf(times, c=shape, scale=scale),
        ),
    ):
        ll = _family_log_likelihood(coordinates, relative, event, family)
        expected = np.log(density[event]).sum() + np.log(survival[~event]).sum()
        np.testing.assert_allclose(ll - event.sum() * offset, expected, rtol=2e-13, atol=2e-13)


def test_family_cdfs_match_guide_parameterizations():
    times = np.array([0.5, 2.0, 5.0])
    relative, offset = _relative_log_times(times)
    shape, scale = 1.8, 2.2
    logs = np.array([np.log(shape)])
    centered_scale = np.array([np.log(scale) - offset])
    for family, expected in (
        ("gamma", gamma.cdf(times, a=shape, scale=scale)),
        ("inverse_gamma", invgamma.cdf(times, a=shape, scale=scale)),
        ("log_logistic", fisk.cdf(times, c=shape, scale=scale)),
    ):
        got = _family_cdf(logs[:, None], centered_scale[:, None], relative, family)[0]
        np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)


def test_gamma_underflow_log_survival_retains_finite_likelihood():
    # scipy's Q underflows here, while the log survival remains finite.
    relative = np.array([np.log(2000.0)])
    event = np.array([False])
    coordinates = np.array([np.log(2.0), 0.0])
    ll = _family_log_likelihood(coordinates, relative, event, "gamma")
    assert np.isfinite(ll) and ll < -1000


def test_tiny_shape_cdf_and_survival_do_not_round_underflowed_argument_to_zero():
    shape = 0.001
    log_x = -1000.0
    relative = np.array([log_x])
    event = np.array([False])
    coordinates = np.array([np.log(shape), 0.0])
    log_lower = shape * log_x - (gammaln(shape) + np.log(shape))
    expected_lower = np.exp(log_lower)
    expected_upper = -np.expm1(log_lower)
    gamma_survival = _family_log_likelihood(coordinates, relative, event, "gamma")
    assert np.isclose(np.exp(gamma_survival), expected_upper, rtol=2e-13)
    gamma_cdf = _family_cdf(coordinates[:1, None], np.array([[0.0]]), relative, "gamma")
    inverse_gamma_cdf = _family_cdf(
        coordinates[:1, None], np.array([[0.0]]), -relative, "inverse_gamma"
    )
    assert np.isclose(gamma_cdf[0, 0], expected_lower, rtol=2e-13)
    assert np.isclose(inverse_gamma_cdf[0, 0], expected_upper, rtol=2e-13)


def test_large_shape_gamma_event_density_uses_stable_near_mode_factor():
    shape = 1.0e12
    # Gamma(shape, scale=1) at its mode x=shape; its log density is
    # -0.5*log(2*pi*shape) + O(1/shape).
    log_x = np.log(shape)
    ll = _family_log_likelihood(
        np.array([np.log(shape), 0.0]), np.array([log_x]), np.array([True]), "gamma"
    )
    expected = -0.5 * np.log(2.0 * np.pi * shape)
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, expected, atol=2e-8, rtol=0)
    inverse_gamma_ll = _family_log_likelihood(
        np.array([np.log(shape), 0.0]),
        np.array([-log_x]),
        np.array([True]),
        "inverse_gamma",
    )
    np.testing.assert_allclose(
        inverse_gamma_ll,
        expected + 2.0 * log_x,
        atol=2e-8,
        rtol=0,
    )


def test_right_censored_fit_has_no_unspecified_johnson_diagnostic():
    result = inverse_gamma_bayesian_gof(
        [0.0, 1.0, 2.0, 3.0],
        event=[False, True, False, True],
        prior_mean=[np.log(2.0), 0.0],
        prior_covariance=[[0.08, 0.01], [0.01, 0.08]],
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(12),
    )
    assert result.diagnostic is None
    assert result.parameters.shape == (2, 8, 2)
    assert np.all(np.isfinite(result.log_likelihood))


def test_complete_fit_returns_paired_cdf_diagnostic():
    result = log_logistic_bayesian_gof(
        [0.8, 1.1, 2.2, 3.0],
        prior_mean=[np.log(1.5), 0.0],
        prior_covariance=[[0.08, 0.0], [0.0, 0.08]],
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(24),
    )
    assert result.diagnostic is not None
    assert result.diagnostic.statistic.shape == (16,)


def test_all_zero_right_censored_data_leaves_the_explicit_prior_unchanged():
    result = gamma_bayesian_gof(
        [0.0, 0.0, 0.0],
        event=[False, False, False],
        prior_mean=[0.0, 0.0],
        prior_covariance=[[0.1, 0.02], [0.02, 0.1]],
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(31),
    )
    np.testing.assert_array_equal(result.log_likelihood, 0.0)
    assert result.diagnostic is None


def test_unit_rescaling_and_budget_rejection_are_stable_before_rng_use():
    times = np.array([0.55, 0.9, 1.4, 2.2, 3.1])
    mean = np.log([2.2, 1.1])
    covariance = np.array([[0.16, 0.055], [0.055, 0.25]])
    base = gamma_bayesian_gof(
        times,
        prior_mean=mean,
        prior_covariance=covariance,
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(19),
    )
    shifted = gamma_bayesian_gof(
        times * np.exp(200.0),
        prior_mean=mean + [0.0, 200.0],
        prior_covariance=covariance,
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(19),
    )
    np.testing.assert_allclose(base.parameters, shifted.parameters, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        base.diagnostic.statistic, shifted.diagnostic.statistic, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        shifted.log_likelihood - base.log_likelihood,
        -times.size * 200.0,
        atol=2e-10,
        rtol=0,
    )

    rng = np.random.default_rng(23)
    state = repr(rng.bit_generator.state)
    with np.testing.assert_raises(ValueError):
        gamma_bayesian_gof(
            times,
            prior_mean=mean,
            prior_covariance=covariance,
            draws=8,
            warmup=0,
            chains=2,
            max_likelihood_evaluations=1,
            rng=rng,
        )
    assert repr(rng.bit_generator.state) == state
