from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit

from mdanderson_stats import (
    fit_phase12_model,
    phase12_response_loglikelihood,
    phase12_response_probabilities,
    phase12_snapshot,
)


def test_source_model_and_independently_observed_calendar_outcomes():
    # Last coefficient is the +/- dose contrast for every adjacent pair.
    b = [0, 0, 0, 1.5]
    expected = np.tile(expit([-1.5, 1.5]), 3)
    assert_allclose(phase12_response_probabilities(b), expected)
    records = [[0, 0, 1, 5, 0, 10], [0, 2, 0, 12, 1, 4], [1, 20, 1, 21, 1, 21]]
    snap = phase12_snapshot(records, time=5)
    assert_allclose(snap.tally[0], [0, 1, 0, 1])
    assert_allclose(snap.enrolled, [2, 0, 0, 0, 0, 0])
    assert_allclose(snap.pending_efficacy, [1, 0, 0, 0, 0, 0])
    assert_allclose(snap.pending_toxicity, [1, 0, 0, 0, 0, 0])
    assert_allclose(phase12_response_loglikelihood(b, snap.tally), np.log(expected[0]))
    late = phase12_snapshot(records, time=12)
    assert_allclose(late.tally[0], [1, 1, 1, 1])
    tally = np.zeros((6, 4))
    tally[1, 1] = 1
    assert_allclose(phase12_response_loglikelihood([0, 0, 0, 1000], tally), 0, atol=1e-100)
    tally[1, 0] = 1
    assert_allclose(phase12_response_loglikelihood([0, 0, 0, 1000], tally), -1000)


def test_full_posterior_against_independent_r_importance_and_prior_recovery():
    tally = np.column_stack((20 - np.arange(1, 7), np.arange(1, 7), np.full(6, 9), np.ones(6)))
    fit = fit_phase12_model(
        tally, draws=3000, warmup=1000, chains=4, rng=np.random.default_rng(8522)
    )
    ref = np.loadtxt(Path(__file__).parent / "fixtures/phase12-model-r.csv", delimiter=",")
    mean = np.r_[fit.coefficient_summary.mean, fit.response_summary.mean]
    se = np.r_[fit.coefficient_summary.batch_mean_mcse, fit.response_summary.batch_mean_mcse]
    assert np.all(abs(mean - ref[:, 0]) < 6 * np.hypot(se, ref[:, 1]))
    assert np.max(fit.coefficient_summary.split_rhat) < 1.04
    assert np.max(fit.response_summary.split_rhat) < 1.04
    assert_allclose(np.diag(fit.pairwise_superiority), 0)
    assert_allclose(
        fit.pairwise_superiority + fit.pairwise_superiority.T, np.ones((6, 6)) - np.eye(6)
    )
    assert fit.reference_superiority[0] == 0.5
    assert np.all(fit.future_probability >= fit.efficacy_probability)
    prior = fit_phase12_model(
        np.zeros((6, 4)), draws=2000, warmup=0, chains=4, rng=np.random.default_rng(8523)
    )
    assert_allclose(prior.coefficient_summary.mean, [6.2445, 2.0815, 2.0815, 0], atol=0.13)
    assert_allclose(prior.coefficient_summary.standard_deviation, 3.16227766, atol=0.12)
    assert prior.likelihood_evaluations == 0
    assert not fit.coefficients.flags.writeable
