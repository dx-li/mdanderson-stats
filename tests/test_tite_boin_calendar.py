"""Accrual resumption at minimum follow-up and at an earlier toxicity event."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import BOINDesign, run_tite_boin_trial, simulate_tite_boin


def test_resume_at_minimum_followup_before_next_outcome():
    gaps = [0, 0, 80, 15, 15, 15]
    potential = np.full((6, 2), np.inf)
    trial = run_tite_boin_trial(BOINDesign(target=0.3), gaps, potential, 90)
    assert [step.time for step in trial.steps] == [95, 102.5]
    assert [step.decision.action for step in trial.steps] == ["suspend_followup", "escalate"]
    assert_array_equal(trial.enrollment_times, [0, 0, 80, 102.5, 117.5, 132.5])
    assert_array_equal(trial.assigned_doses, [1, 1, 1, 2, 2, 2])
    assert trial.suspension_time == 7.5
    assert trial.final_time == 222.5
    scaled = run_tite_boin_trial(
        BOINDesign(target=0.3), np.array(gaps) * 1e-100, potential, 90e-100
    )
    assert_allclose(scaled.enrollment_times / 1e-100, trial.enrollment_times)


def test_toxicity_before_followup_release_changes_next_assignment_only():
    potential = np.full((6, 2), np.inf)
    baseline = run_tite_boin_trial(BOINDesign(target=0.3), [0, 0, 80, 15, 15, 15], potential, 90)
    potential[2, 0] = 20  # DLT at day 100, unobserved at the initial day-95 decision.
    trial = run_tite_boin_trial(BOINDesign(target=0.3), [0, 0, 80, 15, 15, 15], potential, 90)
    assert trial.steps[0].decision.action == baseline.steps[0].decision.action
    assert_allclose(
        trial.steps[0].decision.estimate.estimated_rate,
        baseline.steps[0].decision.estimate.estimated_rate,
    )
    assert trial.steps[1].time == 100
    assert trial.steps[1].decision.action == "stay"
    assert trial.suspension_time == 5
    assert_array_equal(trial.assigned_doses, [1] * 6)
    assert_array_equal(trial.toxicities, [1, 0])


def test_deterministic_simulation_extremes_and_final_followup():
    design = BOINDesign(target=0.3)
    safe = simulate_tite_boin(design, [0, 0], 90, 1 / 15, cohorts=2, trials=5, rng=129)
    assert_array_equal(safe.patients, np.tile([3, 3], (5, 1)))
    assert_array_equal(safe.toxicities, np.zeros((5, 2)))
    assert_array_equal(safe.duration, np.full(5, 240))
    assert_array_equal(safe.selection_probability, [0, 0, 1])
    toxic = simulate_tite_boin(design, [1, 1], 90, 1 / 15, cohorts=2, trials=20, rng=129)
    assert np.all(toxic.selected_dose == 0)
    # With delayed outcomes, the enrollment cap can precede the final safety finding.
    assert set(toxic.stop_reason) <= {"stop_safety", "max_patients"}
    immediate = run_tite_boin_trial(design, [15] * 6, np.zeros((6, 2)), 90)
    assert immediate.stop_reason == "stop_safety"
    assert np.all(toxic.patients[:, 1] == 0)
    assert np.all(toxic.toxicities == toxic.patients)
