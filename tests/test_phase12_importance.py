"""Focused checks for the bounded six-dose adaptive importance backend."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.parallel_phase12_decision import (
    phase12_source_decision,
    phase12_source_final_selection,
)
from mdanderson_stats.parallel_phase12_importance import fit_phase12_importance
from mdanderson_stats.parallel_phase12_model import (
    phase12_response_loglikelihood,
    phase12_response_probabilities,
)


def test_empty_tally_recovers_prior_normalizer_and_is_replayable():
    tally = np.zeros((6, 4))
    first = fit_phase12_importance(
        tally,
        max_integrations=100,
        rng=np.random.default_rng(7201),
    )
    replay = fit_phase12_importance(
        tally,
        max_integrations=100,
        rng=np.random.default_rng(7201),
    )
    assert_allclose(first.posterior_mode, [6.2445, 2.0815, 2.0815, 0.0], atol=0, rtol=0)
    assert_allclose(np.diag(first.proposal_covariance), np.square(3.16227766), rtol=2e-12)
    assert abs(first.log_evidence) < 2e-12
    assert first.integrations == 100
    assert first.converged == replay.converged
    assert_allclose(
        first.response_probability_mean, replay.response_probability_mean, atol=0, rtol=0
    )
    assert_allclose(first.ratio_mc_se, replay.ratio_mc_se, atol=0, rtol=0)
    assert np.all(
        np.abs(first.reference_superiority - 0.5) <= 6 * first.reference_superiority_mc_se + 0.01
    )
    assert_allclose(np.diag(first.pairwise_superiority), 0)
    assert_allclose(
        first.pairwise_superiority + first.pairwise_superiority.T,
        np.ones((6, 6)) - np.eye(6),
        atol=0,
    )
    assert np.all(first.response_probability_second_moment <= first.response_probability_mean)
    assert first.response_probability_mean.shape == (6,)
    assert first.pairwise_superiority.shape == (6, 6)
    assert not first.response_probability_mean.flags.writeable
    assert not first.proposal_covariance.flags.writeable


def test_importance_fit_is_usable_by_source_decision_rules():
    tally = np.zeros((6, 4))
    tally[:, 0] = 6
    tally[:, 1] = [1, 2, 3, 4, 5, 6]
    fit = fit_phase12_importance(
        tally,
        efficacy_target=0.3,
        future_target=0.1,
        max_integrations=200,
        rng=np.random.default_rng(7202),
    )
    decision = phase12_source_decision(
        fit,
        np.full(6, 6),
        phase_one_admissible=np.ones(6, dtype=bool),
        cohort_size=5,
    )
    assert decision.probability.shape == (6,)
    assert phase12_source_final_selection(
        fit, closed=decision.closed, suspended=decision.suspended
    ) in (None, *range(6))


def test_informative_response_means_agree_with_independent_r_importance():
    tally = np.column_stack((20 - np.arange(1, 7), np.arange(1, 7), np.full(6, 9), np.ones(6)))
    fit = fit_phase12_importance(
        tally,
        max_integrations=10_000,
        rng=np.random.default_rng(8524),
    )
    reference = np.loadtxt(Path(__file__).parent / "fixtures/phase12-model-r.csv", delimiter=",")
    discrepancy = abs(fit.response_probability_mean - reference[4:, 0])
    combined_mc_se = np.hypot(fit.response_probability_mc_se, reference[4:, 1])
    assert np.all(discrepancy < 8 * combined_mc_se + 0.003)


def test_native_work_limits_and_tail_mixture_are_checked_before_rng_use():
    rng = np.random.default_rng(7203)
    state = repr(rng.bit_generator.state)
    for kwargs in (
        {"max_integrations": 101},
        {"max_integrations": 1_000_100},
        {"mixture_multivariate": 1.0},
    ):
        try:
            fit_phase12_importance(np.zeros((6, 4)), rng=rng, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid importance settings were accepted: {kwargs}")
    assert repr(rng.bit_generator.state) == state


def test_reported_ratio_mcse_matches_direct_paired_importance_residual():
    tally = np.zeros((6, 4))
    tally[:, 0] = [8, 7, 6, 5, 4, 3]
    tally[:, 1] = [2, 3, 4, 5, 6, 7]
    seed = 7204
    fit = fit_phase12_importance(
        tally,
        integration_relative_error=1e-12,
        max_integrations=100,
        rng=np.random.default_rng(seed),
    )

    rng = np.random.default_rng(seed)
    mode, covariance = fit.posterior_mode, fit.proposal_covariance
    chol = np.linalg.cholesky(covariance)
    prior_mean = np.array([6.2445, 2.0815, 2.0815, 0.0])
    prior_sd = np.full(4, 3.16227766)
    log_weights = []
    values = []
    for _ in range(fit.integrations):
        if rng.random() < 0.99:
            beta = mode + chol @ rng.normal(size=4)
        else:
            beta = rng.normal(prior_mean, prior_sd)
        log_prior = -0.5 * np.sum(((beta - prior_mean) / prior_sd) ** 2)
        log_prior -= np.log(prior_sd).sum() + 2 * np.log(2 * np.pi)
        log_mvn = -2 * np.log(2 * np.pi) - np.log(np.diag(chol)).sum()
        centered = np.linalg.solve(chol, beta - mode)
        log_mvn -= np.dot(centered, centered) / 2
        log_q = np.logaddexp(np.log(0.99) + log_mvn, np.log(0.01) + log_prior)
        log_weights.append(float(phase12_response_loglikelihood(beta, tally) + log_prior - log_q))
        values.append(float(phase12_response_probabilities(beta)[0]))
    weights = np.exp(np.asarray(log_weights) - max(log_weights))
    response = np.asarray(values)
    mean_weight = float(weights.mean())
    estimate = float(np.dot(weights, response) / weights.sum())
    m2_weight = float(np.sum((weights - mean_weight) ** 2))
    weighted_response = weights * response
    m2_response = float(np.sum((weighted_response - weighted_response.mean()) ** 2))
    cross = float(np.sum((weights - mean_weight) * (weighted_response - weighted_response.mean())))
    n = fit.integrations
    expected_se = (
        np.sqrt(
            max(m2_response + estimate**2 * m2_weight - 2 * estimate * cross, 0.0) / (n * (n - 1))
        )
        / mean_weight
    )
    assert fit.integrations == 100
    assert fit.response_probability_mean[0] == pytest.approx(estimate, rel=1e-13, abs=1e-13)
    assert fit.response_probability_mc_se[0] == pytest.approx(expected_se, rel=1e-10, abs=1e-13)
