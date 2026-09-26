"""Focused retrospective snapshot and CRM routing checks."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats.bmacrm import BMACRMPosterior, fit_bmacrm
from mdanderson_stats.bmacrm_decision import BMACRMDecision, bmacrm_decision
from mdanderson_stats.bmacrm_lookahead import BMACRMLookAhead
from mdanderson_stats.crm_calendar import crm_calendar_decision, crm_calendar_snapshot
from mdanderson_stats.dacrm import DACRMPosterior, DACRMPrior
from mdanderson_stats.dacrm_decision import DACRMDecision


def test_snapshot_replays_event_completion_and_pending_boundaries():
    snapshot = crm_calendar_snapshot(
        [0, 0, 1, 1, 2],
        [0, 1, 2, 4, 8],
        [0.5, np.inf, 2, np.inf, 0],
        window=3,
        at=4,
        dose_count=3,
    )
    assert snapshot.row_indices.tolist() == [0, 1, 2, 3]
    assert snapshot.outcomes.tolist() == [1, 0, 1, -1]
    assert snapshot.times.tolist() == [0.5, 3, 2, 0]
    assert snapshot.treated_counts.tolist() == [2, 2, 0]
    assert snapshot.observed_counts.tolist() == [2, 1, 0]
    assert snapshot.toxicities.tolist() == [1, 1, 0]
    assert snapshot.pending_counts.tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        snapshot.outcomes.setflags(write=True)


def test_future_records_cannot_change_an_as_of_snapshot():
    early = crm_calendar_snapshot(
        [0, 1, 0], [0, 1, 20], [0.5, np.inf, 2], window=3, at=2, dose_count=2
    )
    changed_future = crm_calendar_snapshot(
        [0, 1, 1], [0, 1, 20], [0.5, np.inf, 0], window=3, at=2, dose_count=2
    )
    assert_array_equal(early.row_indices, changed_future.row_indices)
    assert_array_equal(early.outcomes, changed_future.outcomes)
    assert_array_equal(early.times, changed_future.times)


def test_snapshot_rejects_unrepresentable_pending_calendar_age():
    with pytest.raises(ArithmeticError, match="rescale"):
        crm_calendar_snapshot([0], [1e308], [np.inf], window=1, at=1e308, dose_count=1)


def test_complete_dacrm_request_routes_to_deterministic_bcrm_without_rng_use():
    snapshot = crm_calendar_snapshot([0, 1], [0, 1], [0.5, np.inf], window=1, at=2, dose_count=2)
    prior = DACRMPrior([0, 1], [1], [1])
    rng = np.random.default_rng(99)
    state_before = rng.bit_generator.state
    routed = crm_calendar_decision(
        snapshot,
        [0.1, 0.3],
        target=0.25,
        method="dacrm",
        da_prior=prior,
        minimum_observed=0,
        rng=rng,
        safety_cutoff=1,
    )
    direct_posterior = fit_bmacrm(
        [0.1, 0.3],
        snapshot.toxicities,
        snapshot.observed_counts,
        target=0.25,
        prior_sd=prior.alpha_sd,
        max_evaluations=200_000,
    )
    direct_decision = bmacrm_decision(
        direct_posterior,
        current_dose=1,
        safety_cutoff=1,
    )
    assert routed.routing == "complete"
    assert isinstance(routed.posterior, BMACRMPosterior)
    assert isinstance(routed.decision, BMACRMDecision)
    assert (routed.decision.action, routed.decision.dose) == (
        direct_decision.action,
        direct_decision.dose,
    )
    assert rng.bit_generator.state == state_before


def test_snapshot_classifies_event_and_completion_at_calendar_endpoints():
    snapshot = crm_calendar_snapshot(
        [0, 1], [0.4, 0.4], [0.1, np.inf], window=0.1, at=0.5, dose_count=2
    )
    assert snapshot.outcomes.tolist() == [1, 0]
    assert snapshot.times.tolist() == [0.1, 0.1]


def test_pending_bmacrm_uses_exact_lookahead_and_pending_dacrm_uses_rng():
    snapshot = crm_calendar_snapshot([0], [0], [np.inf], window=1, at=0, dose_count=2)
    bma = crm_calendar_decision(
        snapshot,
        [0.1, 0.3],
        target=0.25,
        method="bmacrm",
        safety_cutoff=1,
    )
    assert bma.routing == "lookahead"
    assert isinstance(bma.posterior, BMACRMPosterior)
    assert isinstance(bma.decision, BMACRMLookAhead)
    assert bma.evaluations == bma.posterior.evaluations + bma.decision.evaluations

    prior = DACRMPrior([0, 1], [1], [1])
    da = crm_calendar_decision(
        snapshot,
        [0.1, 0.3],
        target=0.25,
        method="dacrm",
        da_prior=prior,
        minimum_observed=0,
        rng=np.random.default_rng(101),
        draws=8,
        warmup=0,
        chains=2,
        safety_cutoff=1,
    )
    assert da.routing == "data_augmentation"
    assert isinstance(da.posterior, DACRMPosterior)
    assert isinstance(da.decision, DACRMDecision)
