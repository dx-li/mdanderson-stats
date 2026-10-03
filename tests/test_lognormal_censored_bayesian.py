import numpy as np
import pytest

from mdanderson_stats.lognormal_censored_bayesian import (
    lognormal_right_censored_bayesian_fit,
)


def _prior():
    return dict(
        prior_location=0.3,
        prior_location_precision=2.0,
        prior_variance_shape=4.5,
        prior_variance_scale=1.1,
    )


def test_mixed_right_censor_fit_retains_paired_chains_and_likelihood():
    times = np.array([0.4, 1.2, 1.2, 2.6, 5.0, 0.0, 3.5])
    event = np.array([True, False, False, True, False, False, False])
    fit = lognormal_right_censored_bayesian_fit(
        times,
        event=event,
        **_prior(),
        draws=1200,
        warmup=800,
        chains=4,
        rng=np.random.default_rng(20261031),
    )
    assert fit.centered_location_samples.shape == (4, 1200)
    assert fit.log_variance_samples.shape == (4, 1200)
    assert fit.log_likelihood_samples.shape == (4, 1200)
    assert fit.parameter_summary.mean.shape == (2,)
    assert fit.parameter_summary.batch_mean_mcse.shape == (2,)
    assert fit.parameter_summary.split_rhat.shape == (2,)
    assert fit.event.tolist() == event.tolist()
    assert fit.times.tolist() == times.tolist()
    assert fit.likelihood_evaluations == 4 * 1200
    assert fit.likelihood_work_units == 4 * 1200 * 6
    assert fit.gibbs_updates == 4 * (800 + 1200)
    assert fit.truncated_normal_proposals >= 4 * (800 + 1200) * 4
    assert np.all(np.isfinite(fit.centered_location_samples))
    assert np.all(np.isfinite(fit.log_variance_samples))
    assert np.all(np.isfinite(fit.log_likelihood_samples))
    assert not fit.event.flags.writeable
    assert not fit.log_variance_samples.flags.writeable
    assert times.flags.writeable


def test_time_unit_change_preserves_centered_posterior_draws():
    times = np.array([0.4, 1.2, 1.2, 2.6, 5.0, 0.0, 3.5])
    event = np.array([True, False, False, True, False, False, False])
    kwargs = dict(
        **_prior(),
        event=event,
        draws=256,
        warmup=128,
        chains=2,
    )
    base = lognormal_right_censored_bayesian_fit(
        times, rng=np.random.default_rng(20261032), **kwargs
    )
    unit_factor = 1e150
    shifted_prior = _prior()
    shifted_prior["prior_location"] += np.log(unit_factor)
    shifted = lognormal_right_censored_bayesian_fit(
        times * unit_factor,
        event=event,
        **shifted_prior,
        draws=256,
        warmup=128,
        chains=2,
        rng=np.random.default_rng(20261032),
    )
    np.testing.assert_allclose(
        shifted.centered_location_samples,
        base.centered_location_samples,
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        shifted.log_variance_samples,
        base.log_variance_samples,
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        shifted.log_likelihood_samples,
        base.log_likelihood_samples - np.count_nonzero(event) * np.log(unit_factor),
        rtol=0,
        atol=2e-11,
    )


def test_all_zero_censors_are_exact_independent_prior_draws():
    draws, chains, seed = 64, 3, 813
    fit = lognormal_right_censored_bayesian_fit(
        [0.0, 0.0, 0.0],
        event=np.zeros(3, dtype=bool),
        **_prior(),
        draws=draws,
        warmup=50,
        chains=chains,
        rng=np.random.default_rng(seed),
    )
    reference = np.random.default_rng(seed)
    expected_mu = np.empty((chains, draws))
    expected_logv = np.empty_like(expected_mu)
    for chain in range(chains):
        gamma = reference.gamma(4.5, size=draws)
        normal = reference.standard_normal(draws)
        expected_logv[chain] = np.log(1.1) - np.log(gamma)
        expected_mu[chain] = normal * np.exp(0.5 * expected_logv[chain]) / np.sqrt(2.0)
    assert fit.location_offset == 0.3
    np.testing.assert_allclose(fit.centered_location_samples, expected_mu, rtol=2e-15, atol=1e-15)
    np.testing.assert_array_equal(fit.log_variance_samples, expected_logv)
    np.testing.assert_array_equal(fit.log_likelihood_samples, np.zeros((chains, draws)))
    assert fit.likelihood_evaluations == 0
    assert fit.likelihood_work_units == 0
    assert fit.parameter_draw_work_units == chains * draws
    assert fit.gibbs_updates == 0
    assert fit.truncated_normal_proposals == 0


def test_prior_input_and_work_preflight_happen_before_rng_use():
    times = [0.4, 1.2, 2.6]
    event = [True, False, False]
    rng1, rng2 = np.random.default_rng(22), np.random.default_rng(22)
    with pytest.raises(ValueError, match="actual Boolean"):
        lognormal_right_censored_bayesian_fit(
            times, event=[1, 0, 0], **_prior(), draws=8, chains=2, rng=rng1
        )
    assert rng1.random() == rng2.random()
    with pytest.raises(ValueError, match="positive"):
        lognormal_right_censored_bayesian_fit(
            times,
            event=event,
            **{**_prior(), "prior_variance_scale": 0},
            draws=8,
            chains=2,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
    with pytest.raises(ValueError, match="minimum Gibbs"):
        lognormal_right_censored_bayesian_fit(
            times,
            event=event,
            **_prior(),
            draws=100,
            warmup=100,
            chains=2,
            max_work=1,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
    with pytest.raises(ValueError, match="minimum truncated-normal"):
        lognormal_right_censored_bayesian_fit(
            times,
            event=event,
            **_prior(),
            draws=8,
            chains=2,
            max_truncated_normal_proposals=1,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()
    with pytest.raises(ValueError, match="minimum Gibbs"):
        lognormal_right_censored_bayesian_fit(
            [0.0, 0.0],
            event=[False, False],
            **_prior(),
            draws=8,
            chains=2,
            max_work=1,
            rng=rng1,
        )
    assert rng1.random() == rng2.random()


def test_all_complete_and_all_censored_posterior_propriety_contracts():
    with pytest.raises(ValueError, match="lognormal_complete_data_bayesian_gof"):
        lognormal_right_censored_bayesian_fit(
            [0.5, 1.5],
            event=np.ones(2, dtype=bool),
            **_prior(),
            draws=8,
            chains=2,
            rng=np.random.default_rng(9),
        )
    # The proper NIG prior makes all-censored posterior draws well-defined.
    result = lognormal_right_censored_bayesian_fit(
        [0.7, 1.5, 2.0, 4.0],
        event=np.zeros(4, dtype=bool),
        **_prior(),
        draws=128,
        warmup=64,
        chains=2,
        rng=np.random.default_rng(10),
    )
    assert np.all(np.isfinite(result.log_likelihood_samples))
    assert np.all(np.isfinite(result.parameter_summary.split_rhat))
