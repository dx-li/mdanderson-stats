"""Deterministic dose-policy checks for DA-CRM summaries."""

from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.dacrm import DACRMPrior, fit_dacrm
from mdanderson_stats.dacrm_decision import dacrm_decision


def _posterior(doses, outcomes, means, overdose=None):
    dose_count = len(means)
    result = fit_dacrm(
        np.linspace(0.05, 0.5, dose_count),
        doses,
        outcomes,
        np.where(
            np.asarray(outcomes) == 1,
            0.5,
            np.where(np.asarray(outcomes) == -1, 0.0, 1.0),
        ),
        prior=DACRMPrior([0, 1], [1], [1]),
        target=0.3,
        rng=np.random.default_rng(10),
        draws=8,
        warmup=0,
        chains=2,
    )
    risks = np.zeros(dose_count) if overdose is None else np.asarray(overdose)
    return replace(result, dose_mean=np.asarray(means, dtype=float), overdose_probability=risks)


def test_paper_profile_reproduces_published_adjacent_dose_sequence():
    table = [
        [0.172, 0.185, 0.209, 0.236, 0.264, 0.294],
        [0.315, 0.336, 0.369, 0.407, 0.445, 0.486],
        [0.028, 0.035, 0.050, 0.071, 0.096, 0.128],
        [0.021, 0.026, 0.037, 0.053, 0.071, 0.093],
        [0.080, 0.114, 0.181, 0.268, 0.359, 0.454],
        [0.175, 0.228, 0.320, 0.422, 0.515, 0.603],
        [0.186, 0.240, 0.333, 0.435, 0.528, 0.615],
        [0.171, 0.223, 0.314, 0.415, 0.509, 0.598],
        [0.189, 0.243, 0.337, 0.440, 0.532, 0.619],
    ]
    currents = [0, 1, 0, 1, 2, 3, 2, 2, 2]
    expected = [1, 0, 1, 2, 3, 2, 2, 2, 2]
    results = []
    for means, current in zip(table, currents, strict=True):
        posterior = _posterior(range(6), [0] * 6, means)
        results.append(
            dacrm_decision(posterior, current_dose=current, policy="paper", safety_cutoff=1).dose
        )
    assert results == expected


def test_start_and_strict_lowest_dose_safety_stop():
    empty = _posterior([], [], [0.2, 0.3], [1.0, 1.0])
    assert dacrm_decision(empty, starting_dose=1).action == "start"
    enrolled = _posterior([0], [0], [0.2, 0.3], [0.96, 0.0])
    assert dacrm_decision(enrolled, current_dose=0).action == "treat"
    unsafe = replace(enrolled, overdose_probability=np.array([0.96001, 0.0]))
    stopped = dacrm_decision(unsafe, current_dose=0)
    assert stopped.action == "stop" and stopped.dose is None


def test_suite_waits_for_minimum_then_caps_at_excessive_raw_rate():
    posterior = _posterior([0, 0, 1, 2, 0], [0, 1, 0, 0, -1], [0.05, 0.15, 0.3])
    waited = dacrm_decision(
        posterior,
        current_dose=0,
        policy="crm_suite",
        minimum_observed=3,
        safety_cutoff=1,
    )
    assert waited.action == "wait"
    ready = dacrm_decision(
        posterior,
        current_dose=0,
        policy="crm_suite",
        minimum_observed=2,
        safety_cutoff=1,
    )
    assert ready.action == "treat" and ready.dose == 0 and ready.raw_rate_limited


def test_suite_final_no_skip_and_three_treated_fallback():
    posterior = _posterior([0, 0, 0, 1], [0, 0, 0, 0], [0.05, 0.1, 0.29])
    result = dacrm_decision(
        posterior,
        policy="crm_suite",
        minimum_observed=1,
        safety_cutoff=1,
        final=True,
    )
    assert result.action == "select_mtd"
    assert result.unconstrained_dose == 2
    assert result.no_skip_dose == 2
    assert result.dose == 0


def test_pending_cohort_does_not_block_paper_step_but_gates_suite_by_minimum():
    posterior = _posterior([0], [-1], [0.05, 0.15, 0.3])
    paper = dacrm_decision(posterior, current_dose=0, policy="paper", safety_cutoff=1)
    assert paper.action == "treat" and paper.dose == 1
    waited = dacrm_decision(
        posterior,
        current_dose=0,
        policy="crm_suite",
        minimum_observed=1,
        safety_cutoff=1,
    )
    assert waited.action == "wait" and waited.dose is None
    suite = dacrm_decision(
        posterior,
        current_dose=0,
        policy="crm_suite",
        minimum_observed=0,
        safety_cutoff=1,
    )
    assert suite.action == "treat" and suite.dose == 1


def test_suite_waits_instead_of_skipping_a_lower_untried_dose():
    posterior = _posterior([0, 2], [0, 0], [0.05, 0.1, 0.15, 0.3, 0.5])
    result = dacrm_decision(
        posterior,
        current_dose=2,
        policy="crm_suite",
        minimum_observed=0,
        safety_cutoff=1,
    )
    assert result.action == "wait" and result.dose is None


def test_suite_requires_minimum_and_rejects_invalid_policy():
    posterior = _posterior([0], [0], [0.2])
    with pytest.raises(ValueError, match="explicit minimum_observed"):
        dacrm_decision(posterior, current_dose=0, policy="crm_suite")
    with pytest.raises(ValueError, match="policy"):
        dacrm_decision(posterior, current_dose=0, policy="other")
