"""Conjugacy, independent beta identities, credible-set geometry and trial behavior."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    BetaBinomialPosterior,
    beta_binomial_sequence,
    simulate_beta_binomial,
)


def test_manual_examples_and_associative_updates():
    # BU1BB manual: uniform prior, one success and two failures -> Beta(2, 3).
    posterior = BetaBinomialPosterior().update(1, 2)
    assert posterior.alpha == 2 and posterior.beta == 3
    assert posterior.mean == 0.4
    assert_allclose(posterior.variance, 0.04)
    sequential = posterior.update(7, 4)
    combined = BetaBinomialPosterior().update(8, 6)
    assert sequential.alpha == combined.alpha and sequential.beta == combined.beta
    # BU2BB manual's two arms, sharing the broadcast implementation.
    arms = BetaBinomialPosterior([1, 0.5], [1, 0.5]).update([1, 6], [3, 4])
    assert_allclose(arms.mean, [1 / 3, 6.5 / 11])


def test_distribution_against_exact_integer_beta_identity():
    a, b = np.array([1, 2, 5, 30])[:, None], np.array([8, 3, 2, 4])[:, None]
    p = np.array([0, 0.125, 0.5, 0.875, 1])
    posterior = BetaBinomialPosterior(a, b)
    expected = []
    for aa, bb in zip(a[:, 0], b[:, 0], strict=True):
        n = int(aa + bb - 1)
        expected.append(
            [
                float(
                    sum(
                        Fraction(comb(n, k))
                        * Fraction(float(x)) ** k
                        * (1 - Fraction(float(x))) ** (n - k)
                        for k in range(int(aa), n + 1)
                    )
                )
                for x in p
            ]
        )
    assert_allclose(posterior.cdf(p), expected, atol=2e-15)
    assert_allclose(posterior.cdf(p) + posterior.sf(p), 1, atol=2e-15)
    assert_allclose(BetaBinomialPosterior(2, 3).pdf(p), 12 * p * (1 - p) ** 2)
    assert_allclose(
        BetaBinomialPosterior(1, 4).quantile(p),
        1 - (1 - p) ** 0.25,
        atol=2e-15,
    )


@pytest.mark.parametrize("a,b", [(2, 3), (50, 7), (0.5, 0.5), (0.1, 0.3), (0.01, 0.01)])
def test_highest_density_mass_and_geometry(a, b):
    posterior = BetaBinomialPosterior(a, b)
    sets = posterior.credible_set([0.5, 0.8, 0.95])
    for bounds, complements, mass in zip(
        sets.intervals, sets.complements, sets.probability, strict=True
    ):
        used = bounds[np.isfinite(bounds[:, 0])]
        reflected = BetaBinomialPosterior(b, a)
        achieved = (
            float(posterior.cdf(used[0, 1]) + reflected.cdf(complements[1, 0]))
            if a < 1 and b < 1
            else sum(float(posterior.cdf(hi) - posterior.cdf(lo)) for lo, hi in used)
        )
        assert_allclose(achieved, mass, atol=2e-10)
        if a < 1 and b < 1:
            x, y = used[0, 1], used[1, 0]
            assert x < y
            assert posterior.logpdf((x + y) / 2) < posterior.logpdf(x)
        else:
            x, y = used[0]
            mode = (a - 1) / (a + b - 2)
            assert x < mode < y
            assert posterior.logpdf(mode) > posterior.logpdf(x)
        right_logpdf = (
            reflected.logpdf(complements[1, 0]) if a < 1 and b < 1 else posterior.logpdf(y)
        )
        assert_allclose(posterior.logpdf(x), right_logpdf, atol=2e-8)


def test_analytic_monotone_uniform_and_arcsine_sets():
    mass = np.array([0.1, 0.5, 0.95])
    uniform = BetaBinomialPosterior().credible_set(mass)
    assert_allclose(uniform.intervals[:, 0, 0], (1 - mass) / 2)
    assert_allclose(uniform.intervals[:, 0, 1], (1 + mass) / 2)
    decreasing = BetaBinomialPosterior(1, 5).credible_set(mass)
    assert_allclose(decreasing.intervals[:, 0, 0], 0)
    assert_allclose(decreasing.intervals[:, 0, 1], 1 - (1 - mass) ** 0.2)
    increasing = BetaBinomialPosterior(5, 1).credible_set(mass)
    assert_allclose(increasing.intervals[:, 0, 0], (1 - mass) ** 0.2)
    assert_allclose(increasing.intervals[:, 0, 1], 1)
    arcsine = BetaBinomialPosterior(0.5, 0.5).credible_set(mass)
    endpoint = np.sin(np.pi * mass / 4) ** 2
    assert_allclose(arcsine.intervals[:, 0, 1], endpoint)
    assert_allclose(arcsine.intervals[:, 1, 0], 1 - endpoint)


def test_complements_preserve_extreme_endpoints_and_equal_tails():
    posterior = BetaBinomialPosterior(0.01, 1000)
    result = posterior.credible_set(0.5)
    assert result.intervals[0, 1] > 0
    reflected = BetaBinomialPosterior(1000, 0.01).credible_set(0.5)
    assert reflected.intervals[0, 0] == 1  # cannot resolve 1 - tiny in float64
    assert_allclose(reflected.complements[0, 0], result.intervals[0, 1], rtol=1e-12)
    equal = BetaBinomialPosterior(2, 3).credible_set(0.9, method="equal-tail")
    assert_allclose(BetaBinomialPosterior(2, 3).cdf(equal.intervals[0]), [0.05, 0.95])
    with pytest.raises(ArithmeticError):
        BetaBinomialPosterior(0.00001, 1).credible_set(0.5)


def test_sequence_batching_empty_cohorts_and_owned_state():
    s = np.array([[1, 2, 0], [0, 3, 1]])
    result = beta_binomial_sequence(s, 3 - s, alpha=[1, 2], beta=1)
    assert_array_equal(result.posterior.alpha, [[1, 2, 4, 4], [2, 2, 5, 6]])
    assert_array_equal(result.posterior.beta, [[1, 3, 4, 7], [1, 4, 4, 6]])
    assert_allclose(result.observed_rate[:, -1], [3 / 9, 4 / 9])
    s[:] = 0
    assert result.successes[0, 0] == 1
    assert not result.posterior.alpha.flags.writeable
    assert beta_binomial_sequence([], []).posterior.mean == 0.5
    assert np.isnan(beta_binomial_sequence([0], [0]).observed_rate[0])


def test_simulation_moments_and_continuation():
    result = simulate_beta_binomial(0.3, trials=30000, rng=np.random.default_rng(2026))
    totals = result.cumulative_successes[:, -1]
    assert_allclose(totals.mean(), 9, atol=0.05)
    assert_allclose(totals.var(), 6.3, atol=0.15)
    rng = np.random.default_rng(19)
    first = simulate_beta_binomial(0.3, cohorts=4, rng=rng)
    second = simulate_beta_binomial(
        0.3,
        cohorts=6,
        alpha=first.posterior.alpha[:, -1],
        beta=first.posterior.beta[:, -1],
        rng=rng,
    )
    whole = simulate_beta_binomial(0.3, rng=np.random.default_rng(19))
    assert_array_equal(second.posterior.alpha[:, -1], whole.posterior.alpha[:, -1])
    assert_array_equal(second.posterior.beta[:, -1], whole.posterior.beta[:, -1])
    for p in [0, 1]:
        endpoint = simulate_beta_binomial(p, rng=np.random.default_rng(1))
        assert_array_equal(endpoint.successes, np.full((1, 10), 3 * p))


@pytest.mark.parametrize("a,b", [(0, 1), (-1, 2), (np.inf, 1), (1e308, 1e308)])
def test_invalid_prior(a, b):
    with pytest.raises(ValueError):
        BetaBinomialPosterior(a, b)


def test_invalid_counts_and_simulation_do_not_consume_rng():
    with pytest.raises(ValueError):
        BetaBinomialPosterior().update(0.5, 2)
    with pytest.raises(ValueError):
        beta_binomial_sequence([2**52, 2**52], [0, 0])
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        simulate_beta_binomial(0.2, alpha=[1, 2], trials=3, rng=rng)
    assert rng.bit_generator.state == before
