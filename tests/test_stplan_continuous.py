"""Focused mathematical checks for STPLAN continuous power methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2, f, nct, t

from mdanderson_stats.stplan_continuous import (
    stplan_exponential_one_sample_power,
    stplan_exponential_two_sample_power,
    stplan_lognormal_two_sample_power,
    stplan_normal_one_sample_power,
    stplan_normal_two_sample_power,
    stplan_welch_two_sample_power,
)
from mdanderson_stats.stplan_correlation import (
    stplan_correlation_one_sample_power,
    stplan_correlation_two_sample_power,
)


def test_one_sample_normal_matches_stplan_manual_example():
    # STPLAN user manual example: means 100 and 101, sd 10, n 156, alpha .05.
    got = stplan_normal_one_sample_power(1, 10, 156)
    assert_allclose(got, 0.344, atol=0.0006)


def test_equal_variance_normal_is_symmetric_and_log_normal_reduces_to_it():
    expected = stplan_normal_two_sample_power(2, 1.4, 24, 31, sides=2)
    assert_allclose(stplan_normal_two_sample_power(-2, 1.4, 24, 31, sides=2), expected)
    # Equal arithmetic means and common CV imply identical log-scale distributions.
    assert_allclose(stplan_lognormal_two_sample_power(3, 3, 0.7, 24, 31), 0.05)


def test_welch_matches_satterthwaite_noncentral_t():
    n1, n2, sd1, sd2, delta = 18.0, 29.0, 1.2, 2.1, 0.8
    v1, v2 = sd1**2 / n1, sd2**2 / n2
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    nc = delta / np.sqrt(v1 + v2)
    expected = nct.sf(t.isf(0.025, df), df, nc)
    assert_allclose(stplan_welch_two_sample_power(delta, sd1, sd2, n1, n2, sides=2), expected)


def test_exponential_power_matches_native_chi_square_and_f_equations():
    n, ratio, alpha = 12.0, 1.7, 0.05
    critical = chi2.isf(alpha, 2 * n)
    assert_allclose(stplan_exponential_one_sample_power(ratio, n), chi2.sf(critical / ratio, 2 * n))
    n1, n2, m1, m2 = 10.0, 14.0, 2.0, 3.0
    critical_f = f.isf(alpha, 2 * n2, 2 * n1)
    assert_allclose(
        stplan_exponential_two_sample_power(m1, m2, n1, n2),
        f.sf(critical_f / (m2 / m1), 2 * n2, 2 * n1),
    )


def test_correlation_methods_have_baseline_and_increasing_effect_power():
    assert_allclose(stplan_correlation_one_sample_power(0, 0, 40), 0.05)
    assert stplan_correlation_one_sample_power(0, 0.45, 40) > 0.05
    assert_allclose(stplan_correlation_two_sample_power(0.2, 0.2, 40, 40), 0.05)
    assert stplan_correlation_two_sample_power(-0.2, 0.5, 40, 45) > 0.05


def test_invalid_domains_and_broadcast_limit_are_rejected():
    with pytest.raises(ValueError):
        stplan_normal_one_sample_power(1, 0, 20)
    with pytest.raises(ValueError):
        stplan_correlation_one_sample_power(1, 0, 30)
    with pytest.raises(ValueError, match="200000"):
        stplan_exponential_one_sample_power(np.ones((500, 1)), np.ones((1, 500)))
