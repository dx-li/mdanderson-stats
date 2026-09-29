"""Focused checks for explicit CiBolus pseudo-data prior calibration."""

import numpy as np
import pytest

from mdanderson_stats.cibolus import (
    CiBolusObservation,
    CiBolusPrior,
    cibolus_predict,
)
from mdanderson_stats.cibolus_calibration import (
    calibrate_cibolus_prior,
    cibolus_prior_predictive_moments,
)
from mdanderson_stats.cibolus_fit import fit_cibolus

_LOG_THETA = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.03, 0.9, 0.12, 0.25, 0.2])


def _fixed_prior() -> CiBolusPrior:
    return CiBolusPrior(_LOG_THETA, np.zeros(11))


def _grid():
    return [0.2], [0.1], [0.5, 1.0]


def test_prior_predictive_reports_source_moments_and_constant_probability_ess() -> None:
    concentrations, bolus, endpoints = _grid()
    result = cibolus_prior_predictive_moments(
        _fixed_prior(),
        concentrations,
        bolus,
        endpoints,
        draws=8,
        chains=2,
        rng=np.random.default_rng(81),
    )
    expected = cibolus_predict(
        _LOG_THETA,
        concentrations,
        bolus,
        endpoints,
        utility=np.zeros((len(endpoints) + 2, 2)),
    )
    np.testing.assert_allclose(result.joint_probability_mean[0, 0], expected.joint[0, 0])
    np.testing.assert_allclose(result.source_probability_mean[0, 0, 1], expected.response_at_one)
    np.testing.assert_array_equal(result.joint_probability_variance, 0)
    assert result.source_probability_names == (
        "p0",
        "response_at_one",
        "toxicity_at_zero_response",
        "toxicity_at_one_response",
    )
    assert np.all(np.isposinf(result.source_probability_ess))
    assert not result.joint_probability_mean.flags.writeable


def test_response_moments_do_not_inherit_toxicity_cell_roundoff() -> None:
    concentrations, bolus, endpoints = _grid()
    mean = _LOG_THETA.copy()
    sd = np.zeros(11)
    sd[6] = 0.7  # Vary toxicity while keeping every response parameter fixed.
    result = cibolus_prior_predictive_moments(
        CiBolusPrior(mean, sd),
        concentrations,
        bolus,
        endpoints,
        draws=8,
        chains=2,
        rng=np.random.default_rng(812),
    )
    np.testing.assert_array_equal(result.bolus_response_variance, 0.0)
    np.testing.assert_array_equal(result.cumulative_response_variance, 0.0)
    assert np.all(np.isposinf(result.bolus_response_ess))
    assert np.all(np.isposinf(result.cumulative_response_ess))


def test_balanced_pseudodata_calibration_retains_counts_and_replay_seed() -> None:
    concentrations, bolus, endpoints = _grid()
    prediction = cibolus_predict(
        _LOG_THETA,
        concentrations,
        bolus,
        endpoints,
        utility=np.zeros((len(endpoints) + 2, 2)),
    )
    rng = np.random.default_rng(82)
    result = calibrate_cibolus_prior(
        concentrations,
        bolus,
        endpoints,
        prediction.joint,
        repetitions=2,
        patients_per_regimen=2,
        pseudo_prior=_fixed_prior(),
        resulting_sd=np.full(11, 9.0),
        pseudo_draws=8,
        pseudo_warmup=0,
        pseudo_chains=2,
        rng=rng,
        max_total_evaluations=36,
        max_total_work=10_000,
    )
    assert result.joint_counts.shape == (2, 1, 1, 4, 2)
    np.testing.assert_array_equal(result.joint_counts.sum(axis=(-2, -1)), 2)
    np.testing.assert_allclose(result.prior.mean, _LOG_THETA, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(result.prior.sd, np.full(11, 9.0))
    np.testing.assert_allclose(result.replicate_mean_mcse, 0.0, rtol=0, atol=2e-15)
    assert result.joint_counts.dtype == np.int64
    assert result.replicate_seeds.dtype == np.uint64
    assert result.likelihood_evaluations.dtype == np.int64
    assert not result.joint_counts.flags.writeable

    replay = np.random.default_rng(int(result.replicate_seeds[0]))
    expected_counts = replay.multinomial(2, result.joint_probabilities[0, 0].reshape(-1)).reshape(
        (4, 2)
    )
    np.testing.assert_array_equal(result.joint_counts[0, 0, 0], expected_counts)


def test_cumulative_budget_rejection_does_not_consume_generator() -> None:
    concentrations, bolus, endpoints = _grid()
    prediction = cibolus_predict(
        _LOG_THETA,
        concentrations,
        bolus,
        endpoints,
        utility=np.zeros((len(endpoints) + 2, 2)),
    )
    first = np.random.default_rng(83)
    second = np.random.default_rng(83)
    state = first.bit_generator.state
    with pytest.raises(ValueError, match="minimum calibration likelihood calls"):
        calibrate_cibolus_prior(
            concentrations,
            bolus,
            endpoints,
            prediction.joint,
            repetitions=2,
            patients_per_regimen=2,
            pseudo_prior=_fixed_prior(),
            resulting_sd=np.ones(11),
            pseudo_draws=8,
            pseudo_warmup=0,
            pseudo_chains=2,
            rng=first,
            max_total_evaluations=35,
        )
    assert first.bit_generator.state == state
    np.testing.assert_array_equal(first.integers(0, 100, size=5), second.integers(0, 100, size=5))


def test_fit_supports_the_paper_sized_400_observation_dataset() -> None:
    concentrations, bolus, endpoints = _grid()
    observations = [CiBolusObservation(0.2, 0.1, "failure", False) for _ in range(400)]
    fit = fit_cibolus(
        observations,
        _fixed_prior(),
        concentrations,
        bolus,
        endpoints,
        utility=np.zeros((len(endpoints) + 2, 2)),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(84),
        max_evaluations=100,
        max_work=100_000,
    )
    assert fit.log_parameters.shape == (2, 8, 11)
    assert fit.likelihood_evaluations >= 18
    assert np.all(np.isfinite(fit.log_likelihood))
