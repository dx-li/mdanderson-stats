"""Exact, bounded pending-outcome look-ahead checks for BMA-CRM."""

import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.bmacrm import fit_bmacrm
from mdanderson_stats.bmacrm_decision import bmacrm_decision
from mdanderson_stats.bmacrm_lookahead import bmacrm_lookahead


def _fit(events=(0, 0, 0), subjects=(3, 0, 0), model_prior=None):
    return fit_bmacrm(
        [0.05, 0.2, 0.5],
        events,
        subjects,
        target=0.3,
        model_prior=model_prior,
        max_evaluations=200_000,
    )


def test_no_pending_delegates_to_complete_outcome_decision():
    posterior = _fit()
    expected = bmacrm_decision(posterior, current_dose=0, safety_cutoff=1)
    result = bmacrm_lookahead(posterior, [0, 0, 0], current_dose=0, safety_cutoff=1)
    assert (result.action, result.dose) == (expected.action, expected.dose)
    assert result.reason == "complete"
    assert result.evaluations == 0
    assert result.evaluated_completions == result.total_completions == 1
    assert result.completion_events.tolist() == [[0, 0, 0]]


def test_all_completion_patterns_agree_before_lookahead_recommends():
    posterior = _fit()
    result = bmacrm_lookahead(
        posterior,
        [0, 0, 2],
        current_dose=0,
        safety_cutoff=1,
        max_completions=3,
    )
    assert result.reason == "invariant"
    assert result.action == "treat" and result.dose == 1
    assert result.total_completions == result.evaluated_completions == 3
    assert result.completion_events.tolist() == [[0, 0, 0], [0, 0, 2], [0, 0, 1]]


def test_outcome_dependent_completion_returns_wait_with_distinguishing_witnesses():
    posterior = _fit()
    result = bmacrm_lookahead(
        posterior,
        [0, 1, 0],
        current_dose=0,
        safety_cutoff=1,
    )
    assert result.reason == "outcome_dependent"
    assert result.action == "wait" and result.dose is None
    assert result.evaluated_completions == 2
    assert [(decision.action, decision.dose) for decision in result.decisions] == [
        ("treat", 2),
        ("treat", 1),
    ]


def test_all_pending_enrollment_is_not_mistaken_for_an_initial_start():
    posterior = _fit(events=(0, 0, 0), subjects=(0, 0, 0))
    result = bmacrm_lookahead(
        posterior,
        [1, 0, 0],
        current_dose=0,
        safety_cutoff=1,
    )
    assert result.reason == "outcome_dependent"
    assert result.action == "wait"
    assert all(decision.action != "start" for decision in result.decisions)


def test_completion_work_limit_returns_wait_without_fitting():
    posterior = _fit()
    result = bmacrm_lookahead(
        posterior,
        [129, 0, 0],
        current_dose=0,
        max_completions=128,
    )
    assert result.reason == "work_limit"
    assert result.action == "wait" and result.dose is None
    assert result.total_completions == 130
    assert result.evaluated_completions == result.evaluations == 0
    assert result.completion_events.shape == (0, 3)
    with pytest.raises(RuntimeError, match="integration exceeded"):
        bmacrm_lookahead(
            posterior,
            [0, 1, 0],
            current_dose=0,
            safety_cutoff=1,
            max_evaluations=1,
        )


def test_original_extreme_model_prior_is_retained_and_reused_for_completions():
    skeletons = [[0.01, 0.99], [0.49, 0.51]]
    prior = [1e-300, 1e300]
    posterior = fit_bmacrm(
        skeletons,
        [0, 4999],
        [5000, 4999],
        target=0.3,
        model_prior=prior,
        max_evaluations=200_000,
    )
    assert_allclose(posterior.input_model_prior, prior, rtol=1e-14)
    assert posterior.prior_model_weights[0] == 0
    result = bmacrm_lookahead(
        posterior,
        [0, 1],
        current_dose=1,
        safety_cutoff=0.9,
    )
    assert result.reason == "invariant"
    assert result.action == "treat" and result.dose == 0
