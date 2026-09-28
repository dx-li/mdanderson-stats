import numpy as np
import pytest
from scipy.special import ndtr

from mdanderson_stats.bayesian_chi_square import bayesian_chi_square_cdf
from mdanderson_stats.lognormal_bayesian_gof import lognormal_complete_data_bayesian_gof


def test_conjugate_parameters_and_paired_cdf_diagnostic() -> None:
    times = np.array([0.7, 1.1, 2.4, 5.0])
    m0, k0, a0, b0 = 0.2, 1.7, 2.5, 0.9
    result = lognormal_complete_data_bayesian_gof(
        times,
        prior_location=m0,
        prior_location_precision=k0,
        prior_variance_shape=a0,
        prior_variance_scale=b0,
        samples=300,
        bins=3,
        rng=104,
    )
    y = np.log(times)
    ybar = float(np.mean(y))
    kn = k0 + times.size
    expected_location = (k0 * m0 + times.size * ybar) / kn
    expected_shape = a0 + times.size / 2
    expected_scale = (
        b0 + np.sum((y - ybar) ** 2) / 2 + (k0 * times.size * (ybar - m0) ** 2 / (2 * kn))
    )
    assert result.posterior_location == pytest.approx(expected_location, rel=1e-14)
    assert result.posterior_location_precision == kn
    assert result.posterior_variance_shape == expected_shape
    assert result.posterior_log_variance_scale == pytest.approx(np.log(expected_scale), rel=1e-14)
    cdf = np.empty((result.centered_location_samples.size, times.size))
    centered_y = np.log(times) - result.location_offset
    for j, (mu, logvar) in enumerate(
        zip(result.centered_location_samples, result.log_variance_samples, strict=True)
    ):
        cdf[j] = ndtr((centered_y - mu) / np.exp(0.5 * logvar))
    expected = bayesian_chi_square_cdf(cdf, bins=3)
    np.testing.assert_array_equal(result.diagnostic.bin_counts, expected.bin_counts)
    np.testing.assert_allclose(result.diagnostic.statistic, expected.statistic, rtol=0, atol=0)

    weak_prior = lognormal_complete_data_bayesian_gof(
        times,
        prior_location=1e300,
        prior_location_precision=1e-20,
        prior_variance_shape=2.5,
        prior_variance_scale=0.9,
        samples=3,
        bins=3,
        rng=17,
    )
    assert weak_prior.posterior_centered_location == pytest.approx(2.5e279, rel=1e-14)


def test_time_unit_invariance_and_rng_preflight() -> None:
    times = np.array([0.5, 1.2, 2.0, 4.0])
    kwargs = dict(
        prior_location=0.3,
        prior_location_precision=2.0,
        prior_variance_shape=3.0,
        prior_variance_scale=1.4,
        samples=250,
        bins=4,
        rng=73,
    )
    base = lognormal_complete_data_bayesian_gof(times, **kwargs)
    shifted = lognormal_complete_data_bayesian_gof(
        times * 1e200,
        **{**kwargs, "prior_location": kwargs["prior_location"] + np.log(1e200)},
    )
    np.testing.assert_allclose(
        shifted.centered_location_samples, base.centered_location_samples, rtol=0, atol=3e-13
    )
    np.testing.assert_allclose(
        shifted.log_variance_samples, base.log_variance_samples, rtol=0, atol=3e-13
    )
    assert shifted.posterior_log_variance_scale == pytest.approx(
        base.posterior_log_variance_scale, abs=3e-13
    )
    np.testing.assert_allclose(
        shifted.diagnostic.statistic, base.diagnostic.statistic, rtol=0, atol=0
    )

    rng1, rng2 = np.random.default_rng(44), np.random.default_rng(44)
    with pytest.raises(ValueError, match="workspace"):
        lognormal_complete_data_bayesian_gof(
            times,
            prior_location=0,
            prior_location_precision=1,
            prior_variance_shape=1,
            prior_variance_scale=1,
            samples=100_000,
            bins=1000,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
