"""BMA-CRM model-selection and Occam-window aggregation checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bmacrm import fit_bmacrm
from mdanderson_stats.bmacrm_decision import bmacrm_decision
from mdanderson_stats.bmacrm_lookahead import bmacrm_lookahead
from mdanderson_stats.crm_calendar import crm_calendar_decision, crm_calendar_snapshot


def test_bms_ties_use_first_model_and_occam_cutoff_is_strict():
    identical = [[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]]
    distinct = [[0.1, 0.2, 0.4], [0.2, 0.3, 0.5]]
    bms = fit_bmacrm(
        distinct,
        [0, 0, 0],
        [0, 0, 0],
        target=0.25,
        aggregation="bms",
    )
    assert_array_equal(bms.posterior_model_weights, [0.5, 0.5])
    assert_array_equal(bms.aggregation_model_weights, [1, 0])
    assert_array_equal(bms.dose_mean, bms.model_dose_mean[0])

    occam = fit_bmacrm(
        identical,
        [0, 0, 0],
        [0, 0, 0],
        target=0.25,
        model_prior=[1, 0.5],
        aggregation="occam",
        occam_threshold=0.5,
    )
    assert_allclose(occam.posterior_model_weights, [2 / 3, 1 / 3])
    assert_array_equal(occam.aggregation_model_weights, [1, 0])
    zero_window = fit_bmacrm(
        identical,
        [0, 0, 0],
        [0, 0, 0],
        target=0.25,
        model_prior=[1, 0],
        aggregation="occam",
        occam_threshold=0,
    )
    assert_array_equal(zero_window.aggregation_model_weights, [1, 0])


def test_lookahead_refit_uses_original_prior_and_can_restore_excluded_model():
    skeletons = [[0.01, 0.05, 0.1], [0.2, 0.3, 0.4]]
    posterior = fit_bmacrm(
        skeletons,
        [0, 0, 0],
        [5, 5, 5],
        target=0.2,
        model_prior=[1, 1],
        aggregation="occam",
        occam_threshold=0.6,
    )
    assert_array_equal(posterior.aggregation_model_weights, [1, 0])
    lookahead = bmacrm_lookahead(
        posterior,
        [0, 0, 1],
        current_dose=2,
        safety_cutoff=1,
        max_completions=2,
    )
    assert_array_equal(lookahead.completion_events, [[0, 0, 0], [0, 0, 1]])
    refits = [
        fit_bmacrm(
            skeletons,
            [0, 0, completed],
            [5, 5, 6],
            target=0.2,
            model_prior=posterior.input_model_prior,
            aggregation="occam",
            occam_threshold=0.6,
        )
        for completed in (0, 1)
    ]
    assert refits[0].aggregation_model_weights[1] == 0
    assert refits[1].aggregation_model_weights[1] > 0
    for decision, refit in zip(lookahead.decisions, refits, strict=True):
        expected = bmacrm_decision(refit, current_dose=2, safety_cutoff=1)
        assert (decision.action, decision.dose) == (expected.action, expected.dose)


def test_aggregation_routes_through_calendar_trial_and_simulation():
    skeletons = [[0.02, 0.05, 0.15, 0.6], [0.05, 0.2, 0.25, 0.35]]
    doses = np.repeat([0, 1, 2], 3)
    delays = np.full(9, np.inf)
    delays[6] = 0.5
    snapshot = crm_calendar_snapshot(doses, np.zeros(9), delays, window=1, at=1, dose_count=4)
    bma = crm_calendar_decision(
        snapshot,
        skeletons,
        target=0.15,
        safety_cutoff=1,
        final=True,
        aggregation="bma",
    )
    bms = crm_calendar_decision(
        snapshot,
        skeletons,
        target=0.15,
        safety_cutoff=1,
        final=True,
        aggregation="bms",
    )
    assert bma.posterior.aggregation == "bma"
    assert bms.posterior.aggregation == "bms"
    assert bma.decision.dose == 1
    assert bms.decision.dose == 2
    assert snapshot.toxicities.tolist() == [0, 0, 1, 0]
    assert snapshot.observed_counts.tolist() == [3, 3, 3, 0]
