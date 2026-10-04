import numpy as np
import pytest
from scipy.stats import chi2

from mdanderson_stats import (
    bayesian_chi_square_cdf,
    exponential_bayesian_gof,
)
from mdanderson_stats.bayesian_chi_square import bayesian_chi_square_discrete_cdf


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


def test_discrete_cdf_mass_randomization_matches_hand_binning_and_replays():
    left = np.array([[0.00, 0.10, 0.24, 0.40, 0.50, 0.74], [0.20, 0.24, 0.25, 0.70, 0.90, 0.10]])
    right = np.array([[0.20, 0.40, 0.30, 0.60, 0.80, 0.90], [0.40, 0.30, 0.75, 0.90, 1.00, 0.90]])
    seed = 1234
    uniforms = np.random.default_rng(seed).random(left.shape)
    randomized = left + uniforms * (right - left)
    randomized = np.maximum(randomized, np.nextafter(left, right))
    expected_counts = np.zeros((left.shape[0], 4), dtype=int)
    edges = (0.25, 0.5, 0.75)
    for row_index, row in enumerate(randomized):
        for value in row:
            bin_index = 0
            while bin_index < len(edges) and value > edges[bin_index]:
                bin_index += 1
            expected_counts[row_index, bin_index] += 1
    expected_statistic = np.sum((expected_counts - 1.5) ** 2 / 1.5, axis=1)

    result = bayesian_chi_square_discrete_cdf(left, right, bins=4, rng=seed)
    replay = bayesian_chi_square_discrete_cdf(left, right, bins=4, rng=seed)
    np.testing.assert_array_equal(result.bin_counts, expected_counts)
    np.testing.assert_allclose(result.statistic, expected_statistic, rtol=0, atol=0)
    np.testing.assert_array_equal(result.bin_counts, replay.bin_counts)
    np.testing.assert_array_equal(result.statistic, replay.statistic)
    assert result.degrees_of_freedom == 3


def test_discrete_cdf_boundaries_and_invalid_preflight_do_not_consume_generator():
    edge = 0.5
    left = np.array([[np.nextafter(edge, 0.0), edge]])
    right = np.array([[edge, np.nextafter(edge, 1.0)]])
    boundary = bayesian_chi_square_discrete_cdf(left, right, bins=2, rng=5)
    np.testing.assert_array_equal(boundary.bin_counts, [[1, 1]])

    valid_left = np.array([[0.1, 0.2]])
    for invalid_left, invalid_right in (
        (valid_left, np.array([[0.4, 0.2]])),
        ([[0.1, [0.2]]], [[0.4, 0.5]]),
        ([[0.1, 0.2 + 1j]], [[0.4, 0.5]]),
        ([[0.1, True]], [[0.4, 0.5]]),
        (
            np.broadcast_to(np.array([[0.1]]), (1001, 1000)),
            np.broadcast_to(np.array([[0.2]]), (1001, 1000)),
        ),
    ):
        actual, control = np.random.default_rng(8), np.random.default_rng(8)
        with pytest.raises(ValueError):
            bayesian_chi_square_discrete_cdf(invalid_left, invalid_right, bins=4, rng=actual)
        assert actual.random() == control.random()


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


def test_exponential_right_censor_update_and_zero_time_censor():
    times = np.array([0.5, 1.0, 2.5, 0.0])
    event = np.array([True, False, True, False])
    fit = exponential_bayesian_gof(
        times,
        prior_shape=1.25,
        prior_rate=0.75,
        event=event,
        samples=6000,
        rng=771,
    )
    assert fit.posterior_shape == 3.25
    assert np.exp(fit.log_posterior_rate) == pytest.approx(4.75)
    assert fit.diagnostic is None
    rates = np.exp(fit.log_rate_samples)
    assert rates.mean() == pytest.approx(3.25 / 4.75, rel=0.025)


def test_exponential_all_censored_requires_proper_posterior_and_boolean_events():
    fit = exponential_bayesian_gof(
        [0.0, 1.0, 2.0],
        prior_shape=2.0,
        prior_rate=0.5,
        event=np.zeros(3, dtype=bool),
        samples=32,
        rng=4,
    )
    assert fit.posterior_shape == 2.0
    assert fit.diagnostic is None
    with pytest.raises(ValueError, match="positive shape and rate"):
        exponential_bayesian_gof(
            [0.0, 0.0], prior_shape=0, prior_rate=0, event=np.zeros(2, dtype=bool), rng=4
        )
    with pytest.raises(ValueError, match="Boolean"):
        exponential_bayesian_gof([1, 2], prior_shape=1, prior_rate=1, event=[1, 0], rng=4)


def test_exponential_complete_sequence_is_unchanged_and_censored_units_rescale():
    times = np.array([0.5, 1.0, 2.5, 4.0])
    implicit = exponential_bayesian_gof(
        times, prior_shape=1.5, prior_rate=0.75, samples=128, bins=3, rng=401
    )
    explicit = exponential_bayesian_gof(
        times,
        prior_shape=1.5,
        prior_rate=0.75,
        event=np.ones(times.size, dtype=bool),
        samples=128,
        bins=3,
        rng=401,
    )
    np.testing.assert_array_equal(implicit.log_rate_samples, explicit.log_rate_samples)
    np.testing.assert_array_equal(implicit.diagnostic.statistic, explicit.diagnostic.statistic)

    event = np.array([True, False, True, False])
    base = exponential_bayesian_gof(
        times, prior_shape=1.5, prior_rate=0.75, event=event, samples=128, rng=402
    )
    scale = 1e100
    shifted = exponential_bayesian_gof(
        times * scale,
        prior_shape=1.5,
        prior_rate=0.75 * scale,
        event=event,
        samples=128,
        rng=402,
    )
    np.testing.assert_allclose(
        shifted.log_rate_samples + np.log(scale), base.log_rate_samples, rtol=0, atol=2e-13
    )
    assert shifted.diagnostic is None


def test_exponential_diagnostic_budget_rejected_before_random_draws():
    rng1, rng2 = np.random.default_rng(8), np.random.default_rng(8)
    with pytest.raises(ValueError, match="bins"):
        exponential_bayesian_gof(
            [1.0, 2.0],
            prior_shape=1,
            prior_rate=1,
            samples=30000,
            bins=1000,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
