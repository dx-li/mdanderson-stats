import numpy as np
import pytest

from mdanderson_stats.bayesian_chi_square import bayesian_chi_square_cdf
from mdanderson_stats.weibull_unknown_shape_gof import (
    weibull_unknown_shape_bayesian_gof,
)


def test_joint_draws_reconstruct_complete_data_likelihood_and_cdf() -> None:
    times = np.array([0.7, 1.1, 2.4, 5.0])
    result = weibull_unknown_shape_bayesian_gof(
        times,
        prior_mean=[0.0, 0.0],
        prior_covariance=[[0.2, 0.06], [0.06, 0.3]],
        draws=16,
        warmup=4,
        chains=2,
        bins=3,
        rng=np.random.default_rng(45),
    )
    flat = result.parameters.reshape((-1, 2))
    centered_times = np.log(times) - result.log_scale_offset
    cdf = np.empty((flat.shape[0], times.size))
    expected_ll = np.empty(flat.shape[0])
    for index, (log_shape, centered_log_scale) in enumerate(flat):
        shape = np.exp(log_shape)
        log_ratio = centered_times - centered_log_scale
        hazard = np.exp(shape * log_ratio)
        cdf[index] = -np.expm1(-hazard)
        expected_ll[index] = (
            times.size * log_shape
            + shape * np.sum(log_ratio)
            - np.sum(hazard)
            - np.sum(np.log(times))
        )
    expected = bayesian_chi_square_cdf(cdf, bins=3)
    np.testing.assert_allclose(result.log_likelihood.ravel(), expected_ll, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(result.diagnostic.bin_counts, expected.bin_counts)
    np.testing.assert_allclose(result.diagnostic.statistic, expected.statistic, rtol=0, atol=0)
    assert result.parameters.shape == (2, 16, 2)
    assert not result.parameters.flags.writeable


def test_time_unit_change_preserves_centered_posterior_and_diagnostic() -> None:
    times = np.array([0.5, 1.2, 2.0, 4.0])
    mean = np.array([0.15, -0.2])
    covariance = np.array([[0.16, -0.04], [-0.04, 0.25]])
    options = dict(draws=24, warmup=5, chains=2, bins=4)
    base = weibull_unknown_shape_bayesian_gof(
        times,
        prior_mean=mean,
        prior_covariance=covariance,
        rng=np.random.default_rng(73),
        **options,
    )
    multiplier = 1e100
    shifted = weibull_unknown_shape_bayesian_gof(
        times * multiplier,
        prior_mean=[mean[0], mean[1] + np.log(multiplier)],
        prior_covariance=covariance,
        rng=np.random.default_rng(73),
        **options,
    )
    np.testing.assert_allclose(shifted.parameters, base.parameters, rtol=0, atol=3e-13)
    np.testing.assert_allclose(
        shifted.log_likelihood,
        base.log_likelihood - times.size * np.log(multiplier),
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_array_equal(shifted.diagnostic.bin_counts, base.diagnostic.bin_counts)
    np.testing.assert_allclose(
        shifted.diagnostic.statistic, base.diagnostic.statistic, rtol=0, atol=0
    )


def test_preflight_rejects_work_and_invalid_covariance_before_rng_use() -> None:
    times = [1.0, 2.0, 3.0]
    mean = [0.0, 0.0]
    covariance = [[1.0, 0.0], [0.0, 1.0]]
    rng1, rng2 = np.random.default_rng(11), np.random.default_rng(11)
    with pytest.raises(ValueError, match="minimum likelihood work"):
        weibull_unknown_shape_bayesian_gof(
            times,
            prior_mean=mean,
            prior_covariance=covariance,
            draws=8,
            warmup=0,
            chains=2,
            max_work=1,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
    with pytest.raises(ValueError, match="positive definite"):
        weibull_unknown_shape_bayesian_gof(
            times,
            prior_mean=mean,
            prior_covariance=[[1.0, 1.0], [1.0, 1.0]],
            draws=8,
            warmup=0,
            chains=2,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()


def test_right_censor_likelihood_unit_invariance_and_zero_time_censoring() -> None:
    times = np.array([0.0, 0.7, 1.4, 2.8, 4.0])
    event = np.array([False, True, False, True, False])
    mean = np.log([1.4, 1.8])
    covariance = np.array([[0.2, 0.04], [0.04, 0.3]])
    options = dict(draws=24, warmup=5, chains=2, bins=3)
    fit = weibull_unknown_shape_bayesian_gof(
        times,
        event=event,
        prior_mean=mean,
        prior_covariance=covariance,
        rng=np.random.default_rng(154),
        **options,
    )
    assert fit.diagnostic is None
    np.testing.assert_array_equal(fit.event, event)
    shape = np.exp(fit.parameters[..., 0])
    scale = np.exp(fit.parameters[..., 1] + fit.log_scale_offset)
    log_ratio = np.log(times[1:])[None, None, :] - np.log(scale)[:, :, None]
    hazard = np.exp(shape[:, :, None] * log_ratio)
    observed_events = event[1:]
    terms = -hazard
    event_terms = (
        np.log(shape)[:, :, None]
        - np.log(times[1:])[None, None, :]
        + shape[:, :, None] * log_ratio
        - hazard
    )
    terms = np.where(observed_events[None, None, :], event_terms, terms)
    expected = np.sum(terms, axis=-1)
    np.testing.assert_allclose(fit.log_likelihood, expected, rtol=2e-13, atol=2e-13)

    multiplier = 1e80
    shifted = weibull_unknown_shape_bayesian_gof(
        times * multiplier,
        event=event.tolist(),
        prior_mean=[mean[0], mean[1] + np.log(multiplier)],
        prior_covariance=covariance,
        rng=np.random.default_rng(154),
        **options,
    )
    np.testing.assert_allclose(shifted.parameters, fit.parameters, rtol=0, atol=4e-13)
    np.testing.assert_allclose(
        shifted.log_likelihood,
        fit.log_likelihood - np.count_nonzero(event) * np.log(multiplier),
        rtol=0,
        atol=3e-12,
    )
    assert shifted.diagnostic is None

    all_zero = weibull_unknown_shape_bayesian_gof(
        [0.0, 0.0, 0.0],
        event=[False, False, False],
        prior_mean=mean,
        prior_covariance=covariance,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(155),
    )
    np.testing.assert_array_equal(all_zero.log_likelihood, 0.0)
    assert all_zero.diagnostic is None
