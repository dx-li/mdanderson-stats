import numpy as np
import pytest

from mdanderson_stats.tte_family_bayesian_gof import log_odds_rate_bayesian_gof


def _prior() -> tuple[np.ndarray, np.ndarray]:
    mean = np.log([1.6, 1.2, 0.7])
    covariance = np.array([[0.13, 0.035, -0.018], [0.035, 0.19, 0.04], [-0.018, 0.04, 0.16]])
    return mean, covariance


def test_log_odds_rate_complete_fit_retains_three_paired_parameters_and_johnson_result():
    mean, covariance = _prior()
    result = log_odds_rate_bayesian_gof(
        [0.45, 0.8, 1.3, 2.1, 3.4],
        prior_mean=mean,
        prior_covariance=covariance,
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(28731),
    )

    assert result.parameter_names == ("log_shape", "centered_log_scale", "log_c")
    assert result.parameters.shape == (2, 8, 3)
    assert result.log_likelihood.shape == (2, 8)
    assert result.parameter_summary.mean.shape == (3,)
    assert result.diagnostic is not None
    assert result.diagnostic.statistic.shape == (16,)


def test_log_odds_rate_censored_fit_omits_unspecified_johnson_result():
    mean, covariance = _prior()
    result = log_odds_rate_bayesian_gof(
        [0.45, 0.8, 1.3, 2.1, 3.4],
        event=[True, False, True, True, False],
        prior_mean=mean,
        prior_covariance=covariance,
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(73182),
    )

    assert result.diagnostic is None
    assert result.parameters.shape == (2, 8, 3)
    assert np.all(np.isfinite(result.log_likelihood))


def test_log_odds_rate_time_unit_shift_changes_event_loglikelihood_only():
    times = np.array([0.45, 0.8, 1.3, 2.1, 3.4])
    events = np.array([True, False, True, True, False])
    mean, covariance = _prior()
    options = dict(
        event=events,
        prior_covariance=covariance,
        draws=8,
        warmup=2,
        chains=2,
    )
    base = log_odds_rate_bayesian_gof(
        times,
        prior_mean=mean,
        rng=np.random.default_rng(1197),
        **options,
    )
    shift = 200.0
    transformed_mean = mean.copy()
    transformed_mean[1] += shift
    changed_units = log_odds_rate_bayesian_gof(
        times * np.exp(shift),
        prior_mean=transformed_mean,
        rng=np.random.default_rng(1197),
        **options,
    )

    np.testing.assert_allclose(base.parameters, changed_units.parameters, atol=2e-11, rtol=0)
    np.testing.assert_allclose(
        changed_units.log_likelihood - base.log_likelihood,
        -events.sum() * shift,
        atol=1e-10,
        rtol=0,
    )


def test_log_odds_rate_invalid_work_budget_fails_before_rng_use():
    mean, covariance = _prior()
    rng = np.random.default_rng(2381)
    state_before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="minimum chain work"):
        log_odds_rate_bayesian_gof(
            [0.45, 0.8, 1.3, 2.1, 3.4],
            prior_mean=mean,
            prior_covariance=covariance,
            draws=8,
            warmup=2,
            chains=2,
            max_work=1,
            rng=rng,
        )
    assert repr(rng.bit_generator.state) == state_before


def test_log_odds_rate_all_zero_censored_data_has_constant_likelihood():
    mean, covariance = _prior()
    result = log_odds_rate_bayesian_gof(
        [0.0, 0.0],
        event=[False, False],
        prior_mean=mean,
        prior_covariance=covariance,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(1894),
    )

    np.testing.assert_array_equal(result.log_likelihood, 0.0)
    assert result.diagnostic is None
