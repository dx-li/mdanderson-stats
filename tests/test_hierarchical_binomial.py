"""Analytic comparison models and independent integration checks for MCMC inference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import expit

from mdanderson_stats.hierarchical_binomial import hierarchical_binomial, summarize_chains


def test_manual_independent_and_pooled_models_and_reproducibility():
    settings = dict(
        successes=[2, 3, 5, 1, 2], trials=[10, 10, 10, 5, 3], draws=40, warmup=20, chains=2
    )
    settings["independent_prior"] = (0.001, 0.001)
    one = hierarchical_binomial(**settings, rng=np.random.default_rng(106))
    two = hierarchical_binomial(**settings, rng=np.random.default_rng(106))
    assert_allclose(one.pooled.alpha, 13.001)
    assert_allclose(one.pooled.beta, 25.001)
    assert_allclose(one.independent.alpha, np.array(settings["successes"]) + 0.001)
    assert_array_equal(one.group_logit, two.group_logit)
    assert_array_equal(one.precision, two.precision)
    assert one.group_logit.shape == (2, 40, 5)
    assert not one.group_logit.flags.writeable
    assert_allclose(one.overall_probability, expit(one.global_logit))


def test_group_posterior_against_independent_quadrature():
    # Concentrate both hyperpriors around mu=0, tau=1. This approaches separate
    # logistic-normal posteriors whose moments can be integrated independently.
    result = hierarchical_binomial(
        [2, 8],
        [10, 10],
        prior_mean_precision=1e12,
        precision_shape=1e12,
        precision_rate=1e12,
        draws=4000,
        warmup=500,
        chains=4,
        rng=np.random.default_rng(19),
    )
    summary = summarize_chains(result.group_probability)
    for i, y in enumerate([2, 8]):

        def density(z):
            return np.exp(-0.5 * z * z - y * np.logaddexp(0, -z) - (10 - y) * np.logaddexp(0, z))

        norm = quad(density, -12, 12, epsabs=1e-12)[0]
        mean = quad(lambda z: expit(z) * density(z), -12, 12, epsabs=1e-12)[0] / norm
        assert abs(float(summary.mean[i]) - mean) < max(0.006, 5 * summary.batch_mean_mcse[i])
    assert np.all(summary.split_rhat < 1.03)


def test_no_data_groups_retain_gaussian_prior_in_fixed_hyperparameter_limit():
    result = hierarchical_binomial(
        [0, 0],
        [0, 0],
        prior_mean_precision=1e12,
        precision_shape=1e12,
        precision_rate=1e12,
        draws=3000,
        warmup=200,
        rng=np.random.default_rng(61),
    )
    assert_allclose(result.group_logit.mean(axis=(0, 1)), 0, atol=0.05)
    assert_allclose(result.group_logit.var(axis=(0, 1)), 1, atol=0.08)


def test_summary_detects_separated_chains_and_interval_coverage():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 1000, 2))
    summary = summarize_chains(x)
    assert np.all(summary.split_rhat < 1.02)
    assert np.all(summary.batch_mean_mcse > 0)
    empirical = np.mean((x >= summary.interval[:, 0]) & (x <= summary.interval[:, 1]), axis=(0, 1))
    assert_allclose(empirical, 0.8, atol=1 / x.shape[0] / x.shape[1])
    x[0] += 10
    assert np.all(summarize_chains(x).split_rhat > 2)
    assert np.all(np.isnan(summarize_chains(np.ones((2, 8))).split_rhat))


def test_invalid_inputs_do_not_consume_random_state():
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    for args in [
        dict(successes=[1, 2], trials=[1, 1]),
        dict(successes=[1], trials=[2]),
        dict(successes=[1, 1], trials=[2, 2], precision_shape=0),
    ]:
        with pytest.raises(ValueError):
            hierarchical_binomial(**args, rng=rng)
    assert rng.bit_generator.state == before


def test_full_hierarchy_without_data_recovers_joint_prior():
    fit = hierarchical_binomial(
        [0, 0],
        [0, 0],
        prior_mean_precision=1,
        precision_shape=3,
        precision_rate=3,
        draws=5000,
        warmup=1000,
        rng=np.random.default_rng(8106),
    )
    assert_allclose(fit.global_logit.mean(), 0, atol=0.1)
    assert_allclose(fit.global_logit.var(), 1, atol=0.12)
    assert_allclose(fit.precision.mean(), 1, atol=0.08)
    # Var(theta_i) = Var(mu) + E(1/tau) = 1 + rate/(shape-1).
    assert_allclose(fit.group_logit.var(axis=(0, 1)), 2.5, atol=0.25)
