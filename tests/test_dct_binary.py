import numpy as np
import pytest
from scipy.stats import norm

from mdanderson_stats import dct_binary_sample_size, dct_normal_sample_size


def test_unequal_arm_formula_and_rounded_power():
    result = dct_normal_sample_size(
        10, 20, 25, onsite_experimental_sd=25, offsite_experimental_sd=30, relative_bias=-0.2
    )
    np.testing.assert_array_equal(result.allocation, [[36, 36], [106, 106]])
    rate = 0.25 * 100 / (2 * (400 + 625)) + 0.75 * 64 / (2 * (625 + 900))
    assert result.unrounded_total == pytest.approx((norm.isf(0.025) + norm.ppf(0.8)) ** 2 / rate)
    uneven = dct_normal_sample_size(
        10,
        20,
        25,
        onsite_experimental_sd=25,
        offsite_experimental_sd=30,
        randomization_ratio=1.7,
        relative_bias=-0.2,
    )
    n = uneven.allocation
    shift = np.sqrt(100 / (400 / n[0, 0] + 625 / n[0, 1]) + 64 / (625 / n[1, 0] + 900 / n[1, 1]))
    assert uneven.achieved_power == pytest.approx(
        norm.cdf(shift - norm.isf(0.025)) + norm.cdf(-shift - norm.isf(0.025))
    )
    assert uneven.achieved_power >= 0.8


def test_binary_direct_information_and_repeated_measure_reduction():
    result = dct_binary_sample_size(0.2, 0.4, 0.25, 0.4)
    rate = 0.25 * 0.2**2 / (2 * (0.2 * 0.8 + 0.4 * 0.6)) + 0.75 * 0.15**2 / (
        2 * (0.25 * 0.75 + 0.4 * 0.6)
    )
    assert result.unrounded_total == pytest.approx((norm.isf(0.025) + norm.ppf(0.8)) ** 2 / rate)
    np.testing.assert_array_equal(result.allocation, [[31, 31], [92, 92]])
    repeated = dct_binary_sample_size(
        0.2,
        0.4,
        0.25,
        0.4,
        onsite_repeats=5,
        offsite_repeats=5,
        onsite_correlation=0.5,
        offsite_correlation=0.5,
    )
    assert repeated.unrounded_total == pytest.approx(0.6 * result.unrounded_total)
    assert repeated.achieved_power >= 0.8
    fully = dct_binary_sample_size(0.2, 0.4, 0.2, 0.4, offsite_fraction=1)
    assert fully.unrounded_total == pytest.approx(
        2 * (0.16 + 0.24) / 0.2**2 * (norm.isf(0.025) + norm.ppf(0.8)) ** 2
    )


def test_known_variance_weighted_test_monte_carlo_power():
    # Independent Gaussian stratum mean differences validate the actual test being planned.
    result = dct_binary_sample_size(0.2, 0.4, 0.25, 0.4, randomization_ratio=2, sides=1)
    n = result.allocation
    variance = np.array([0.16 / n[0, 0] + 0.24 / n[0, 1], 0.1875 / n[1, 0] + 0.24 / n[1, 1]])
    effects = np.array([0.2, 0.15])
    rng = np.random.default_rng(164)
    draws = rng.normal(effects, np.sqrt(variance), (100000, 2))
    z = (draws * effects / variance).sum(axis=1) / np.sqrt(np.sum(effects**2 / variance))
    empirical = np.mean(z > norm.isf(0.05))
    assert abs(empirical - result.achieved_power) < 0.005
    with pytest.raises(ValueError):
        dct_binary_sample_size(0, 0.4, 0.2, 0.4)
