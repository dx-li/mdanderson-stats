"""Ordering probabilities checked with finite exact sums and analytic identities."""

from fractions import Fraction
from math import comb, factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import BetaBinomialPosterior, compare_beta_binomial


def exact_ordering(a, b, c, d):
    # Integer beta CDF is a binomial tail. Integrating each polynomial term
    # against the other beta density gives an exact rational beta-moment sum.
    def beta(x, y):
        return Fraction(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))

    n = a + b - 1
    return float(sum(comb(n, k) * beta(c + k, d + n - k) / beta(c, d) for k in range(a, n + 1)))


def test_against_exact_polynomial_integrals():
    cases = [(1, 1, 2, 3), (2, 4, 7, 5), (12, 2, 2, 12), (30, 20, 10, 40), (2, 5, 20, 3)]
    a, b, c, d = np.array(cases).T
    result = compare_beta_binomial(BetaBinomialPosterior(a, b), BetaBinomialPosterior(c, d))
    expected = np.array([exact_ordering(*case) for case in cases])
    assert_allclose(result.treatment_greater, expected, atol=5e-10, rtol=0)
    assert_allclose(result.control_greater, 1 - expected, atol=5e-10, rtol=0)
    assert np.all(result.absolute_error < 1e-9)


def test_manual_example_and_prior_comparison():
    control = BetaBinomialPosterior().update(1, 3)
    treatment = BetaBinomialPosterior(0.5, 0.5).update(6, 4)
    result = compare_beta_binomial(control, treatment)
    # BU2BB help prints 0.8627 (four decimal places).
    assert round(float(result.treatment_greater), 4) == 0.8627
    prior = compare_beta_binomial(BetaBinomialPosterior(), BetaBinomialPosterior(0.5, 0.5))
    assert_allclose(prior.treatment_greater, 0.5, atol=5e-10)


def test_uniform_comparison_is_mean_even_for_singular_beta():
    treatment = BetaBinomialPosterior([0.01, 1000, 0.5, 5], [1000, 0.01, 2, 0.1])
    result = compare_beta_binomial(BetaBinomialPosterior(), treatment)
    assert_allclose(result.treatment_greater, treatment.mean, atol=5e-10, rtol=0)
    assert_allclose(result.control_greater, 1 - treatment.mean, atol=5e-10, rtol=0)


def test_symmetry_reflection_and_concentration():
    left = BetaBinomialPosterior([0.01, 1000], [0.02, 1000])
    right = BetaBinomialPosterior([0.03, 1001], [0.01, 999])
    original = compare_beta_binomial(left, right)
    reflected = compare_beta_binomial(
        BetaBinomialPosterior(right.beta, right.alpha), BetaBinomialPosterior(left.beta, left.alpha)
    )
    assert_allclose(original.treatment_greater, reflected.treatment_greater, atol=1e-9)
    identical = compare_beta_binomial(left, left)
    assert_allclose(identical.treatment_greater, 0.5, atol=0)
    assert np.all(original.treatment_greater > 0.5)


def test_tolerance_and_result_ownership():
    with pytest.raises(ValueError):
        compare_beta_binomial(
            BetaBinomialPosterior(), BetaBinomialPosterior(), absolute_tolerance=0
        )
    result = compare_beta_binomial(BetaBinomialPosterior(), BetaBinomialPosterior(2, 3))
    assert not result.treatment_greater.flags.writeable
