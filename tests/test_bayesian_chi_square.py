import numpy as np
import pytest
from scipy.stats import chi2

from mdanderson_stats import bayesian_chi_square_cdf, exponential_bayesian_gof


def test_bin_endpoints_direct_pearson_statistics_and_reference_summaries():
    result = bayesian_chi_square_cdf([[0, 0.25, 0.5, 0.75, 1], [0.1, 0.1, 0.1, 0.9, 0.9]], bins=4)
    counts = np.array([[2, 1, 1, 1], [3, 0, 0, 2]])
    expected = np.sum((counts - 1.25) ** 2 / 1.25, axis=1)
    np.testing.assert_array_equal(result.bin_counts, counts)
    np.testing.assert_allclose(result.statistic, expected)
    np.testing.assert_allclose(result.reference_tail_probability, chi2.sf(expected, 3))
    assert result.area_against_reference == pytest.approx(chi2.cdf(expected, 3).mean())
    assert result.critical_value == pytest.approx(chi2.ppf(0.95, 3))
    assert result.degrees_of_freedom == 3
    assert not result.statistic.flags.writeable


def test_exact_rate_posterior_and_time_unit_invariance():
    times = np.array([1.0, 2.0, 4.0, 8.0])
    result = exponential_bayesian_gof(times, prior_shape=2, prior_rate=3, samples=20000, rng=66)
    rates = np.exp(result.log_rate_samples)
    assert rates.mean() == pytest.approx(6 / 18, rel=0.02)
    assert rates.var() == pytest.approx(6 / 18**2, rel=0.04)
    for scale in [1e-200, 1e200]:
        scaled = exponential_bayesian_gof(
            times * scale, prior_shape=2, prior_rate=3 * scale, samples=20000, rng=66
        )
        np.testing.assert_allclose(
            scaled.log_rate_samples + np.log(scale), result.log_rate_samples, atol=2e-13
        )
        np.testing.assert_array_equal(scaled.diagnostic.statistic, result.diagnostic.statistic)


def test_joint_data_and_posterior_reference_calibration_and_bad_fit():
    rng = np.random.default_rng(2004)
    # Each row is an independent repeated dataset followed by one posterior draw.
    times = rng.exponential(3, (2000, 500))
    rates = rng.gamma(500, size=2000) / times.sum(axis=1)
    cdf = -np.expm1(-times * rates[:, None])
    repeated = bayesian_chi_square_cdf(cdf, bins=5)
    assert abs(repeated.mean_statistic - 4) < 0.2
    assert abs(repeated.critical_exceedance_fraction - 0.05) < 0.02
    wrong = exponential_bayesian_gof(
        np.linspace(0.1, 2, 500), prior_shape=0, prior_rate=0, samples=1000, bins=5, rng=2004
    )
    assert wrong.diagnostic.critical_exceedance_fraction > 0.95
