"""Focused complete-outcome conduct-rule checks for BMA-CRM."""

import numpy as np
import pytest

from mdanderson_stats.bmacrm import BMACRMPosterior
from mdanderson_stats.bmacrm_decision import bmacrm_decision


def _posterior(means, overdose, events, subjects, target=0.3):
    dose_mean = np.asarray(means, dtype=float)
    event_count = np.asarray(events, dtype=float)
    subject_count = np.asarray(subjects, dtype=float)
    model_means = dose_mean[None, :]
    model_overdose = np.asarray(overdose, dtype=float)[None, :]
    return BMACRMPosterior(
        np.asarray(means, dtype=float)[None, :],
        event_count,
        subject_count,
        target,
        np.sqrt(2),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        model_means,
        model_overdose,
        dose_mean,
        model_overdose[0],
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        0,
    )


def test_initial_start_precedes_safety_and_final_requires_data():
    prior = _posterior([0.1, 0.3, 0.5], [0.99, 0.2, 0.01], [0, 0, 0], [0, 0, 0])
    decision = bmacrm_decision(prior, starting_dose=1, safety_cutoff=0.9)
    assert decision.action == "start"
    assert decision.dose == decision.no_skip_dose == 1
    with pytest.raises(ValueError, match="without observed subjects"):
        bmacrm_decision(prior, final=True)


def test_strict_lowest_dose_safety_rule_stops_but_cutoff_one_disables_it():
    unsafe = _posterior([0.1, 0.3], [0.9, 0.2], [0, 1], [1, 2])
    result = bmacrm_decision(unsafe, current_dose=0, safety_cutoff=0.89)
    assert result.action == "stop" and result.dose is None
    at_boundary = bmacrm_decision(unsafe, current_dose=0, safety_cutoff=0.9)
    assert at_boundary.action == "treat"
    disabled = bmacrm_decision(unsafe, current_dose=0, safety_cutoff=1)
    assert disabled.action == "treat"


def test_nearest_target_uses_lower_tie_and_no_skip_caps_at_first_untried():
    posterior = _posterior([0.1, 0.3, 0.48, 0.7], [0.1] * 4, [0, 0, 0, 0], [1, 0, 1, 0], target=0.5)
    result = bmacrm_decision(posterior, current_dose=0, safety_cutoff=1)
    assert result.unconstrained_dose == 2
    assert result.no_skip_dose == 1
    assert result.dose == 1

    tie = _posterior([0.1, 0.3], [0.1, 0.1], [0, 0], [1, 0], target=0.2)
    assert bmacrm_decision(tie, current_dose=0, safety_cutoff=1).dose == 0


def test_raw_rate_caps_current_or_intervening_escalation_only():
    posterior = _posterior([0.1, 0.2, 0.3, 0.4], [0.1] * 4, [0, 1, 0, 0], [1, 2, 1, 1], target=0.45)
    result = bmacrm_decision(posterior, current_dose=0, safety_cutoff=1)
    assert result.no_skip_dose == 3
    assert result.dose == 1
    assert result.raw_rate_limited
    current_limited = bmacrm_decision(posterior, current_dose=1, safety_cutoff=1)
    assert current_limited.no_skip_dose == 3
    assert current_limited.dose == 1
    assert current_limited.raw_rate_limited


def test_final_three_subject_fallback_and_uncertainty_do_not_apply_raw_cap():
    posterior = _posterior([0.1, 0.2, 0.3, 0.4], [0.1] * 4, [3, 3, 1, 1], [3, 3, 1, 1], target=0.4)
    result = bmacrm_decision(posterior, final=True, safety_cutoff=1)
    assert result.unconstrained_dose == 3
    assert result.dose == 1
    assert not result.high_uncertainty
    assert not result.raw_rate_limited

    no_lower_support = _posterior([0.1, 0.2, 0.4], [0.1] * 3, [0, 0, 1], [1, 1, 1], target=0.4)
    uncertain = bmacrm_decision(no_lower_support, final=True, safety_cutoff=1)
    assert uncertain.dose == 2
    assert uncertain.high_uncertainty
