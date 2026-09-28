"""Focused checks for EffTox prior marginal moments and calibration."""

import numpy as np
import pytest

from mdanderson_stats.efftox_calibration import (
    calibrate_efftox_prior,
    efftox_prior_moments,
)
from mdanderson_stats.efftox_model import EffToxPrior


def test_symmetric_efficacy_predictor_has_half_mean_and_beta_moment_ess():
    prior = EffToxPrior(
        mean=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        sd=[0.7, 0.0, 0.8, 0.5, 0.0, 0.0],
    )
    result = efftox_prior_moments([1, 2, 4], prior, outcome="efficacy", quadrature_order=48)

    np.testing.assert_allclose(result.mean, 0.5, rtol=0, atol=2e-14)
    assert np.all(result.variance > 0)
    np.testing.assert_allclose(
        result.effective_sample_size,
        0.25 / result.variance - 1.0,
        rtol=2e-14,
        atol=0,
    )
    assert not result.mean.flags.writeable


def test_fully_fixed_prior_has_zero_variance_and_infinite_beta_ess():
    prior = EffToxPrior(
        mean=[-2.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        sd=np.zeros(6),
    )
    result = efftox_prior_moments([1, 2, 4], prior, outcome="toxicity")

    np.testing.assert_array_equal(result.variance, 0.0)
    assert np.isposinf(result.effective_sample_size).all()


def test_calibration_reports_distinct_targets_and_optimizer_status():
    result = calibrate_efftox_prior(
        [1, 2, 4, 6.6, 10],
        [0.2, 0.4, 0.6, 0.8, 0.9],
        [0.02, 0.04, 0.06, 0.08, 0.1],
        efficacy_target_ess=0.9,
        toxicity_target_ess=1.0,
        monotone_toxicity=False,
        quadrature_order=40,
        max_iterations=700,
        max_evaluations=1_200,
    )

    assert result.efficacy_target_ess == 0.9
    assert result.toxicity_target_ess == 1.0
    assert result.prior.monotone_toxicity is False
    assert np.all(np.isfinite(result.prior.mean))
    assert np.all(np.isfinite(result.prior.sd))
    assert result.efficacy_optimizer_success
    assert result.toxicity_optimizer_success
    assert result.quadrature_converged
    assert np.isfinite(result.quadrature_error)
    assert result.quadrature_error >= 0.0
    assert result.quadrature_order == 40
    assert result.efficacy.quadrature_order == 40
    assert result.efficacy.mean.shape == (5,)


def test_monotone_and_untruncated_toxicity_are_distinct_model_choices():
    prior = EffToxPrior(
        mean=[-1.0, -0.4, 0.0, 1.0, 0.0, 0.0],
        sd=[0.8, 0.5, 0.8, 0.5, 0.0, 0.0],
        monotone_toxicity=True,
    )
    monotone = efftox_prior_moments([1, 2, 4], prior, outcome="toxicity")
    untruncated_prior = EffToxPrior(prior.mean, prior.sd, monotone_toxicity=False)
    untruncated = efftox_prior_moments([1, 2, 4], untruncated_prior, outcome="toxicity")

    assert monotone.monotone_toxicity
    assert not untruncated.monotone_toxicity
    assert monotone.mean[0] < untruncated.mean[0]
    assert monotone.mean[1] == pytest.approx(untruncated.mean[1], abs=5e-16)
    assert monotone.mean[2] > untruncated.mean[2]

    fixed_intercept = EffToxPrior(
        mean=[-1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        sd=[0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        monotone_toxicity=True,
    )
    conditional = efftox_prior_moments([1, 2, 4], fixed_intercept, outcome="toxicity")
    assert np.all(np.diff(conditional.mean) > 0)
    assert conditional.variance[0] > 0 and conditional.variance[2] > 0
    assert conditional.variance[1] == 0.0
