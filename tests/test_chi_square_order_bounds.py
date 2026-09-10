import numpy as np
import pytest
from scipy.stats import chi2

from mdanderson_stats import bayesian_chi_square_cdf, chi_square_order_bounds


def test_published_twenty_percent_example_and_search_correction():
    threshold = chi2.isf(0.01, 4)
    result = chi_square_order_bounds(np.r_[np.zeros(800), np.full(200, threshold)], 4)
    assert result.minimum_diagnostic == pytest.approx(0.05)
    assert result.minimizing_rank == 801
    assert result.ranks[-1] == 995
    assert result.search_adjusted_bound == 1
    np.testing.assert_allclose(
        result.fixed_rank_bound,
        np.minimum(1, 1000 * chi2.sf(result.statistics, 4) / (1001 - result.ranks)),
    )
    assert not result.fixed_rank_bound.flags.writeable


def test_bound_under_strong_dependence_and_perfect_dependence():
    rng = np.random.default_rng(1995)
    # Each marginal is chi-square(1), with strong dependence between draws.
    normal = np.sqrt(0.9) * rng.normal(size=(30000, 1)) + np.sqrt(0.1) * rng.normal(
        size=(30000, 12)
    )
    ordered = np.sort(normal**2, axis=1)
    threshold = chi2.isf(0.05, 1)
    result = chi_square_order_bounds(np.full(12, threshold), 1, upper_trim=0)
    observed = (ordered > threshold).mean(axis=0)
    np.testing.assert_array_less(observed, result.fixed_rank_bound + 0.008)
    # Perfect dependence makes the minimum-rank inequality sharp.
    assert result.fixed_rank_bound[0] == pytest.approx(0.05)
    assert result.fixed_rank_bound[-1] == pytest.approx(0.6)


def test_trim_extreme_tails_and_posterior_integration():
    result = chi_square_order_bounds([1e6, 1e6], 2, upper_trim=0)
    assert result.minimum_diagnostic > 0
    assert result.search_adjusted_bound >= result.minimum_diagnostic
    diagnostic = bayesian_chi_square_cdf([[0.1, 0.2, 0.9], [0.3, 0.4, 0.5]], bins=3)
    bounds = diagnostic.order_bounds(upper_trim=0)
    np.testing.assert_array_equal(bounds.statistics, np.sort(diagnostic.statistic))
    with pytest.raises(ValueError):
        chi_square_order_bounds([-1], 3)
