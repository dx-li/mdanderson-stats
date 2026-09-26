"""Focused fixed-cohort CRM scheduling and conservation checks."""

import numpy as np

from mdanderson_stats.crm_trial import run_crm_trial

SKELETONS = [0.08, 0.2, 0.4]


def test_trial_enrolls_fixed_cohorts_and_makes_final_mtd_decision():
    result = run_crm_trial(
        SKELETONS,
        [0, 0, 1, 0],
        np.full((4, 3), np.inf),
        1,
        target=0.25,
        cohort_size=2,
        safety_cutoff=1,
    )
    assert result.assigned_doses.shape == (4,)
    assert result.enrollment_times.tolist() == [0, 0, 1, 1]
    assert result.treated_counts.sum() == result.toxicities.sum() + 4
    assert result.selected_dose is not None
    assert result.stop_reason == "max_patients"
    assert result.final_time == 2
    assert result.steps[-1].action == "select_mtd"
    assert result.steps[-1].routing == "complete"


def test_lookahead_wait_suspends_enrollment_without_adding_another_gap():
    delays = np.full((4, 3), np.inf)
    delays[:2, :] = 0.5
    result = run_crm_trial(
        SKELETONS,
        [0, 0, 0.1, 0],
        delays,
        1,
        target=0.25,
        cohort_size=2,
        safety_cutoff=1,
        max_completions=1,
    )
    assert any(step.action == "wait" for step in result.steps)
    assert result.enrollment_times.tolist() == [0, 0, 0.5, 0.5]
    assert result.suspension_time == 0.4
    assert result.treated_counts.tolist() == [4, 0, 0]
    assert result.stop_reason == "max_patients"


def test_early_safety_stop_tracks_decision_and_final_followup_separately():
    delays = np.full((4, 3), np.inf)
    delays[0, 0] = 0.05
    result = run_crm_trial(
        SKELETONS,
        [0, 0, 0.1, 0],
        delays,
        1,
        target=0.25,
        cohort_size=2,
        safety_cutoff=0.5,
    )
    assert result.assigned_doses.size == 2
    assert result.stop_reason == "safety_stop"
    assert result.selected_dose is None
    assert result.decision_time == 0.1
    assert result.final_time == 1
    assert result.toxicities.tolist() == [1, 0, 0]
