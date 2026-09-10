"""Hierarchical normal inference checked against Gaussian algebra and quadrature."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from mdanderson_stats import NormalSample, summarize_chains
from mdanderson_stats.hierarchical_normal import hierarchical_normal

GROUPS = [[2, 6, 7, 3.8, 4.2], [11, 12, 10.7, 6.5, 7.9, 12.3, 15]]


def test_manual_empirical_comparisons_and_raw_summary_equivalence():
    raw = hierarchical_normal(GROUPS, draws=20, warmup=20, chains=2, rng=np.random.default_rng(107))
    assert_allclose(raw.sample.mean, [4.6, 10.771428571428572])
    assert_allclose(raw.sample.sample_variance, [np.var(g, ddof=1) for g in GROUPS])
    joined = np.concatenate(GROUPS)
    assert_allclose(raw.pooled_sample.mean, joined.mean())
    assert_allclose(raw.pooled_sample.sample_variance, joined.var(ddof=1))
    equivalent = hierarchical_normal(
        raw.sample, draws=20, warmup=20, chains=2, rng=np.random.default_rng(107)
    )
    assert_allclose(equivalent.group_mean, raw.group_mean, rtol=1e-13, atol=1e-13)
    assert raw.empirical_density([0, 1]).shape == (2, 2)
    assert raw.empirical_density([0, 1], pooled=True).shape == (2,)
    assert not raw.group_mean.flags.writeable


def test_gaussian_joint_posterior_in_fixed_precision_limit():
    # With tightly concentrated Gamma priors, the joint posterior of (mu,theta)
    # is a multivariate normal whose precision is independently assembled here.
    sample = NormalSample([-1, 2], [4, 7], sample_sd=[1, 1])
    fit = hierarchical_normal(
        sample,
        prior_mean_precision=1,
        observation_shape=1e12,
        observation_rate=1e12,
        between_shape=1e12,
        between_rate=1e12,
        draws=6000,
        warmup=500,
        rng=np.random.default_rng(471),
    )
    precision = np.array([[3, -1, -1], [-1, 5, 0], [-1, 0, 8]], dtype=float)
    covariance = np.linalg.solve(precision, np.eye(3))
    mean = np.linalg.solve(precision, [0, -4, 14])
    draws = np.concatenate((fit.global_mean[..., None], fit.group_mean), axis=-1).reshape(-1, 3)
    assert_allclose(draws.mean(axis=0), mean, atol=0.025)
    assert_allclose(np.cov(draws, rowvar=False), covariance, atol=0.025)
    assert np.all(summarize_chains(fit.group_mean).split_rhat < 1.03)


def test_unknown_observation_precision_against_marginal_integration():
    sample = NormalSample([-1, 1], [2, 2], sample_sd=[1, 1])
    fit = hierarchical_normal(
        sample,
        prior_mean_precision=1e12,
        observation_shape=2,
        observation_rate=2,
        between_shape=1e12,
        between_rate=1e12,
        draws=5000,
        warmup=500,
        rng=np.random.default_rng(701),
    )
    # Integrating phi gives likelihood proportional to
    # (b + (S+n*(theta-ybar)^2)/2)^(-a-n/2).
    for i, observed in enumerate([-1, 1]):

        def rate(theta):
            return 2 + 0.5 * (1 + 2 * (theta - observed) ** 2)

        def density(theta):
            return np.exp(-(theta**2) / 2) * rate(theta) ** -3

        z = quad(density, -12, 12, epsabs=1e-12)[0]
        mean = quad(lambda theta: theta * density(theta), -12, 12, epsabs=1e-12)[0] / z
        phi = quad(lambda theta: 3 / rate(theta) * density(theta), -12, 12, epsabs=1e-12)[0] / z
        assert_allclose(fit.group_mean[..., i].mean(), mean, atol=0.035)
        assert_allclose(fit.observation_precision[..., i].mean(), phi, atol=0.035)


def test_reproducible_draws_and_constant_data():
    options = dict(groups=[[2, 2, 2], [3, 3, 3]], draws=20, warmup=30, chains=2)
    first = hierarchical_normal(**options, rng=np.random.default_rng(107))
    second = hierarchical_normal(**options, rng=np.random.default_rng(107))
    assert_array_equal(first.group_mean, second.group_mean)
    assert np.all(np.isfinite(first.observation_precision))
    with pytest.raises(ValueError):
        first.empirical_density([2, 3])


def test_invalid_input_precedes_random_draws():
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    for groups in ([[1]], [[1, 2], []], NormalSample([1, 2], [3, 3])):
        with pytest.raises(ValueError):
            hierarchical_normal(groups, rng=rng)
    with pytest.raises(ValueError):
        hierarchical_normal(GROUPS, observation_rate=0, rng=rng)
    assert rng.bit_generator.state == before
