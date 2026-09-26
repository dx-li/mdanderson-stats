"""Focused mathematical checks for STPLAN discrete power methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri
from scipy.stats import binom, chi2, ncx2, poisson

from mdanderson_stats.stplan_discrete import (
    stplan_arcsine_binomial_two_sample_power,
    stplan_binomial_k_sample_power,
    stplan_exact_binomial_power,
    stplan_exact_poisson_power,
    stplan_fisher_exact_approx_power,
    stplan_historical_binomial_power,
    stplan_median_split_power,
    stplan_responder_normal_approximation_power,
    stplan_retention_probability,
)


def test_arcsine_binomial_and_median_split_reduce_to_stplan_transform():
    p1, p2, n1, n2, alpha = 0.2, 0.45, 35.0, 42.0, 0.025
    effect = abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))
    expected = ndtr(effect / np.sqrt(1 / n1 + 1 / n2) - ndtri(1 - alpha))
    assert_allclose(stplan_arcsine_binomial_two_sample_power(p1, p2, n1, n2, alpha=alpha), expected)
    assert_allclose(stplan_median_split_power(0.4, 0, 100), 0.05)


def test_historical_control_and_responder_normal_formulas():
    pc, pe, nc, ne, alpha = 0.25, 0.4, 80.0, 45.0, 0.05

    def trans(p):
        return 2 * np.arcsin(np.sqrt(p))

    expected_historical = ndtr(
        (abs(trans(pe) - trans(pc)) + ndtri(alpha) * np.sqrt(1 / ne + 1 / nc)) / np.sqrt(1 / ne)
    )
    assert_allclose(stplan_historical_binomial_power(pc, pe, nc, ne), expected_historical)

    pc, pe, nc, ne, margin, confidence = 0.7, 0.8, 60.0, 50.0, 0.1, 0.95
    variance = pe * (1 - pe) / ne + pc * (1 - pc) / nc
    expected_responder = ndtr((margin - pe + pc) / np.sqrt(variance) - ndtri(confidence))
    assert_allclose(
        stplan_responder_normal_approximation_power(pc, pe, nc, ne, margin, confidence=confidence),
        expected_responder,
    )


def test_k_sample_chi_square_and_native_two_group_one_sided_rule():
    probabilities = np.array([0.2, 0.6])
    sizes = np.array([30.0, 40.0])
    pooled = np.sum(probabilities * sizes) / np.sum(sizes)
    lam = np.sum(sizes * (probabilities - pooled) ** 2) / (pooled * (1 - pooled))
    expected = ncx2.sf(chi2.isf(0.1, 1), 1, lam)
    assert_allclose(stplan_binomial_k_sample_power(probabilities, sizes, sides=1), expected)
    assert_allclose(stplan_binomial_k_sample_power([0.3, 0.3], sizes), 0.05)
    with pytest.raises(ValueError, match="only for two groups"):
        stplan_binomial_k_sample_power([0.2, 0.4, 0.7], [10, 10, 10], sides=1)


def test_retention_probability_matches_binomial_tail_and_handles_zero_duration():
    expected = binom.sf(74, 100, 0.9**3)
    assert_allclose(stplan_retention_probability(100, 75, 0.1, 3), expected)
    assert_allclose(stplan_retention_probability(12, 12, 1, 0), 1)


def test_fisher_method_is_the_named_native_approximation_including_small_n_artifact():
    p1, p2, n, alpha = 0.35, 0.6, 8.0, 0.025
    pooled_sd = np.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2))
    alt_sd = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    z = (abs(n * abs(p2 - p1) - 1) / np.sqrt(n) + ndtri(alpha) * pooled_sd) / alt_sd
    assert_allclose(stplan_fisher_exact_approx_power(p1, p2, n, alpha=alpha), ndtr(z))
    assert stplan_fisher_exact_approx_power(0.4, 0.4, 2) > 0.05


def test_exact_binomial_power_selects_a_discrete_size_limited_region():
    expected = binom.cdf(1, 10, 0.1)
    assert_allclose(stplan_exact_binomial_power(0.5, 0.1, 10), expected)
    assert_allclose(stplan_exact_binomial_power(0.5, 0.5, 1), 0)
    assert_allclose(stplan_exact_binomial_power(0.5, 0.1, 10, alpha=0.1), binom.cdf(2, 10, 0.1))
    assert_allclose(
        stplan_exact_binomial_power(0.5, 0.1, 10, alpha=0.1, sides=2),
        binom.cdf(1, 10, 0.1),
    )


def test_exact_poisson_power_and_empty_rejection_region():
    # Under null mean 1, X >= 4 is the largest upper-tail rejection region <= .05.
    expected = poisson.sf(3, 3)
    assert_allclose(stplan_exact_poisson_power(1, 3, 1), expected)
    assert_allclose(stplan_exact_poisson_power(1, 3, 1, alpha=0.1, sides=2), expected)
    assert_allclose(stplan_exact_poisson_power(1, 0.5, 0.1), 0)


def test_inputs_require_integer_exact_sample_sizes_and_bound_broadcasts():
    with pytest.raises(ValueError, match="integers"):
        stplan_exact_binomial_power(0.4, 0.6, 10.5)
    with pytest.raises(ValueError, match="200000"):
        stplan_arcsine_binomial_two_sample_power(
            np.ones((500, 1)) * 0.3, np.ones((1, 500)) * 0.4, 10, 10
        )
