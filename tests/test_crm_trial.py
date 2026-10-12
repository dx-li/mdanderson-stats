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


def test_native_desktop_enrollment_wait_overshoot_and_terminal_clocks():
    # Original controller contracts: first at zero, retry at candidate arrivals,
    # consume a gap after the final patient; duration is actual ascertainment.
    result = run_crm_trial(
        SKELETONS,
        [0.25, 0.25, 0.75, 0.25, 10],
        np.full((4, 3), np.inf),
        1,
        target=0.25,
        cohort_size=2,
        safety_cutoff=1,
        max_completions=1,
        scheduling="native_desktop",
    )
    assert result.enrollment_times.tolist() == [0, 0.25, 1.25, 1.5]
    assert result.suspension_time == 0.75
    assert result.arrival_gaps_used == 5
    assert result.final_time == 2.5
    assert result.decision_time == 12.5
    assert result.steps[-1].time == 12.5
    assert result.scheduling == "native_desktop"


def test_native_desktop_requires_complete_positive_arrival_tape():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        run_crm_trial(
            SKELETONS,
            [0, 1],
            np.full((2, 3), np.inf),
            1,
            target=0.25,
            cohort_size=2,
            scheduling="native_desktop",
        )
    with pytest.raises(RuntimeError, match="arrival-gap tape"):
        run_crm_trial(
            SKELETONS,
            [0.1] * 4,
            np.full((4, 3), np.inf),
            1,
            target=0.25,
            cohort_size=2,
            scheduling="native_desktop",
            max_completions=1,
        )


def test_original_native_duration_accumulator_fixtures():
    import json
    from pathlib import Path

    cases = json.loads((Path(__file__).parent / "fixtures/crm-native-duration.json").read_text())
    for case in cases:
        patients = np.asarray(case["patient_enrollment_and_final_ascertainment_delay"])
        if not patients.size or patients[0, 0] != 0:
            continue
        gaps = np.r_[np.diff(patients[:, 0]), 100]
        delays = np.repeat(patients[:, 1, None], 3, axis=1)
        result = run_crm_trial(
            SKELETONS,
            gaps,
            delays,
            10,
            target=0.25,
            cohort_size=len(patients),
            safety_cutoff=1,
            scheduling="native_desktop",
        )
        assert result.final_time == case["duration"]
