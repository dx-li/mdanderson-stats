import numpy as np
import pytest

from mdanderson_stats.bayesian_chi_square import exponential_bayesian_gof
from mdanderson_stats.weibull_bayesian_gof import weibull_fixed_shape_bayesian_gof


def test_shape_one_matches_existing_exponential_posterior_exactly() -> None:
    times = np.array([0.4, 1.2, 2.7, 4.1])
    exponential = exponential_bayesian_gof(
        times, prior_shape=1.5, prior_rate=0.7, samples=400, bins=4, rng=1234
    )
    weibull = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=1.0,
        prior_shape=1.5,
        prior_rate=0.7,
        samples=400,
        bins=4,
        rng=1234,
    )
    np.testing.assert_allclose(
        weibull.centered_log_rate_samples + weibull.log_rate_offset,
        exponential.log_rate_samples,
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(weibull.diagnostic.statistic, exponential.diagnostic.statistic)
    assert weibull.posterior_shape == exponential.posterior_shape
    assert weibull.log_posterior_rate == exponential.log_posterior_rate


def test_fixed_shape_gamma_update_and_cdf_identity() -> None:
    times = np.array([0.5, 1.25, 2.0, 3.5])
    shape, prior_shape, prior_rate = 1.7, 2.25, 0.8
    result = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=shape,
        prior_shape=prior_shape,
        prior_rate=prior_rate,
        samples=300,
        bins=3,
        rng=71,
    )
    expected_shape = prior_shape + times.size
    expected_rate = prior_rate + np.sum(times**shape)
    assert result.posterior_shape == expected_shape
    assert result.log_posterior_rate == pytest.approx(np.log(expected_rate), rel=1e-14)
    rates = np.exp(result.centered_log_rate_samples + result.log_rate_offset)
    cdf = -np.expm1(-rates[:, None] * times[None, :] ** shape)
    # Recomputing the public diagnostic from the posterior CDFs is an independent
    # check of the Weibull CDF mapping and preserved observed-data rows.
    from mdanderson_stats.bayesian_chi_square import bayesian_chi_square_cdf

    direct = bayesian_chi_square_cdf(cdf, bins=3)
    np.testing.assert_array_equal(result.diagnostic.bin_counts, direct.bin_counts)
    np.testing.assert_allclose(result.diagnostic.statistic, direct.statistic, rtol=0, atol=0)


def test_time_unit_scaling_and_extreme_log_domain_inputs() -> None:
    times = np.array([0.3, 0.8, 1.6, 3.2])
    beta, prior_shape, prior_rate = 2.0, 1.0, 0.5
    base = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=beta,
        prior_shape=prior_shape,
        prior_rate=prior_rate,
        samples=250,
        bins=4,
        rng=992,
    )
    for scale in (1e-100, 1e100):
        shifted = weibull_fixed_shape_bayesian_gof(
            times * scale,
            weibull_shape=beta,
            prior_shape=prior_shape,
            prior_rate=prior_rate * scale**beta,
            samples=250,
            bins=4,
            rng=992,
        )
        np.testing.assert_allclose(
            shifted.centered_log_rate_samples + shifted.log_rate_offset + beta * np.log(scale),
            base.centered_log_rate_samples + base.log_rate_offset,
            rtol=0,
            atol=3e-13,
        )
        np.testing.assert_array_equal(shifted.diagnostic.statistic, base.diagnostic.statistic)

    extreme = weibull_fixed_shape_bayesian_gof(
        [1e-200, 2e-200, 4e-200],
        weibull_shape=2.0,
        prior_shape=0,
        prior_rate=0,
        samples=64,
        rng=5,
    )
    assert np.all(np.isfinite(extreme.centered_log_rate_samples))
    assert np.all(np.isfinite(extreme.diagnostic.statistic))


def test_cell_budget_and_unrepresentable_power_fail_before_rng() -> None:
    times = np.array([1.0, 2.0])
    rng1, rng2 = np.random.default_rng(29), np.random.default_rng(29)
    with pytest.raises(ValueError, match="workspace"):
        weibull_fixed_shape_bayesian_gof(
            times,
            weibull_shape=1.5,
            prior_shape=1,
            prior_rate=1,
            samples=100_000,
            bins=1000,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()

    with pytest.raises(ArithmeticError, match=r"log\(times\)"):
        weibull_fixed_shape_bayesian_gof(
            [1e300, 2e300],
            weibull_shape=1e308,
            prior_shape=1,
            prior_rate=1,
            samples=10,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()


def test_centered_rate_retains_gamma_variation_for_huge_identical_times() -> None:
    result = weibull_fixed_shape_bayesian_gof(
        np.full(500, 1e100),
        weibull_shape=1e305,
        prior_shape=0,
        prior_rate=0,
        samples=5000,
        bins=10,
        rng=812,
    )
    scaled_rate_draws = np.exp(result.centered_log_rate_samples)
    # With identical t, lambda*t**beta has distribution Gamma(n, 1) / n.
    hazards = scaled_rate_draws * result.posterior_rate_scaled_sum / 500
    assert np.mean(hazards) == pytest.approx(1.0, rel=0.06)
    assert np.var(hazards, ddof=1) == pytest.approx(1 / 500, rel=0.08)
    assert np.all(np.isfinite(result.diagnostic.statistic))


def test_centered_time_powers_resolve_adjacent_huge_times() -> None:
    xmax = 1e100
    times = np.array([np.nextafter(xmax, 0.0), xmax])
    result = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=1e305,
        prior_shape=0,
        prior_rate=0,
        samples=128,
        bins=4,
        rng=91,
    )
    hazards = np.exp(result.centered_log_rate_samples)
    cdf = np.column_stack((np.zeros(hazards.size), -np.expm1(-hazards)))
    from mdanderson_stats.bayesian_chi_square import bayesian_chi_square_cdf

    expected = bayesian_chi_square_cdf(cdf, bins=4)
    np.testing.assert_array_equal(result.diagnostic.bin_counts, expected.bin_counts)
    np.testing.assert_allclose(result.diagnostic.statistic, expected.statistic, rtol=0, atol=0)


def test_fixed_shape_weibull_right_censor_update_and_zero_time_censor() -> None:
    times = np.array([0.5, 1.0, 2.5, 0.0])
    event = np.array([True, False, True, False])
    beta, prior_shape, prior_rate = 1.5, 1.25, 0.75
    result = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=beta,
        prior_shape=prior_shape,
        prior_rate=prior_rate,
        event=event,
        samples=6000,
        rng=771,
    )
    expected_rate = prior_rate + np.sum(times**beta)
    assert result.posterior_shape == prior_shape + event.sum()
    assert np.exp(result.log_posterior_rate) == pytest.approx(expected_rate)
    assert result.diagnostic is None
    hazards = np.exp(result.centered_log_rate_samples + result.log_rate_offset) * expected_rate
    assert hazards.mean() == pytest.approx(result.posterior_shape, rel=0.025)


def test_fixed_shape_weibull_all_censored_and_zero_exposure_prior() -> None:
    all_censored = np.zeros(3, dtype=bool)
    result = weibull_fixed_shape_bayesian_gof(
        [0.0, 1.0, 2.0],
        weibull_shape=2.0,
        prior_shape=2.0,
        prior_rate=0.5,
        event=all_censored,
        samples=32,
        rng=22,
    )
    assert result.posterior_shape == 2.0
    assert result.log_posterior_rate == pytest.approx(np.log(5.5))
    assert result.diagnostic is None

    zero_censor = weibull_fixed_shape_bayesian_gof(
        [0.0, 0.0],
        weibull_shape=1.7,
        prior_shape=2.0,
        prior_rate=0.5,
        event=np.zeros(2, dtype=bool),
        samples=32,
        rng=23,
    )
    assert zero_censor.log_posterior_rate == pytest.approx(np.log(0.5))
    with pytest.raises(ValueError, match="positive shape and rate"):
        weibull_fixed_shape_bayesian_gof(
            [0.0, 0.0],
            weibull_shape=1.7,
            prior_shape=0,
            prior_rate=0,
            event=np.zeros(2, dtype=bool),
            rng=23,
        )


def test_fixed_shape_weibull_complete_sequence_and_censored_units():
    times = np.array([0.5, 1.0, 2.5, 4.0])
    common = dict(
        weibull_shape=1.5,
        prior_shape=1.5,
        prior_rate=0.75,
        samples=128,
        bins=3,
        rng=401,
    )
    implicit = weibull_fixed_shape_bayesian_gof(times, **common)
    explicit = weibull_fixed_shape_bayesian_gof(
        times, event=np.ones(times.size, dtype=bool), **common
    )
    np.testing.assert_array_equal(
        implicit.centered_log_rate_samples, explicit.centered_log_rate_samples
    )
    np.testing.assert_array_equal(implicit.diagnostic.statistic, explicit.diagnostic.statistic)

    event = np.array([True, False, True, False])
    base = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=1.5,
        prior_shape=1.5,
        prior_rate=0.75,
        event=event,
        samples=128,
        rng=402,
    )
    scale = 1e100
    shifted = weibull_fixed_shape_bayesian_gof(
        times * scale,
        weibull_shape=1.5,
        prior_shape=1.5,
        prior_rate=0.75 * scale**1.5,
        event=event,
        samples=128,
        rng=402,
    )
    np.testing.assert_allclose(
        shifted.centered_log_rate_samples + shifted.log_rate_offset + 1.5 * np.log(scale),
        base.centered_log_rate_samples + base.log_rate_offset,
        rtol=0,
        atol=3e-13,
    )
    assert shifted.diagnostic is None
