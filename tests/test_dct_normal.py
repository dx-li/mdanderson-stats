import numpy as np
import pytest
from scipy.stats import norm

from mdanderson_stats import dct_normal_sample_size


def test_paper_equations_and_correlated_help_examples():
    independent = dct_normal_sample_size(10, 20, 25, relative_bias=-0.2)
    formula = 4 * (norm.isf(0.025) + norm.ppf(0.8)) ** 2 / 0.5**2 * 4 / (1 + 3 * 0.8**2 / 1.5625)
    assert independent.unrounded_total == pytest.approx(formula)
    np.testing.assert_array_equal(independent.allocation, [[29, 29], [85, 85]])
    repeated = dct_normal_sample_size(
        10,
        20,
        25,
        relative_bias=-0.2,
        onsite_repeats=5,
        offsite_repeats=5,
        onsite_correlation=0.5,
        offsite_correlation=0.5,
    )
    assert repeated.unrounded_total == pytest.approx(formula * 0.6)
    np.testing.assert_array_equal(repeated.allocation, [[17, 17], [51, 51]])
    assert repeated.achieved_power >= 0.8


def test_full_decentralization_and_direct_rounded_information():
    result = dct_normal_sample_size(
        10, 20, 20, offsite_fraction=1, offsite_repeats=5, offsite_correlation=0.5
    )
    np.testing.assert_array_equal(result.allocation, [[0, 0], [38, 38]])
    variance = 20**2 * 0.6 * (1 / 38 + 1 / 38)
    shift = 10 / np.sqrt(variance)
    assert result.achieved_power == pytest.approx(
        norm.cdf(shift - norm.isf(0.025)) + norm.cdf(-shift - norm.isf(0.025))
    )
    unequal = dct_normal_sample_size(10, 20, 30, randomization_ratio=2, relative_bias=-0.3)
    n = unequal.allocation
    information = 10**2 / (400 * (1 / n[0, 0] + 1 / n[0, 1])) + 7**2 / (
        900 * (1 / n[1, 0] + 1 / n[1, 1])
    )
    assert unequal.achieved_power == pytest.approx(
        norm.cdf(np.sqrt(information) - norm.isf(0.025))
        + norm.cdf(-np.sqrt(information) - norm.isf(0.025))
    )


def test_scale_invariance_and_correlation_limits():
    base = dct_normal_sample_size(10, 20, 25)
    for scale in [1e-200, 1e200]:
        result = dct_normal_sample_size(10 * scale, 20 * scale, 25 * scale)
        assert result.unrounded_total == pytest.approx(base.unrounded_total, rel=1e-12)
        np.testing.assert_array_equal(result.allocation, base.allocation)
    identical = dct_normal_sample_size(
        10,
        20,
        25,
        onsite_repeats=100,
        offsite_repeats=100,
        onsite_correlation=1,
        offsite_correlation=1,
    )
    assert identical.unrounded_total == pytest.approx(base.unrounded_total)
    with pytest.raises(ValueError):
        dct_normal_sample_size(10, 20, 25, relative_bias=-1)
