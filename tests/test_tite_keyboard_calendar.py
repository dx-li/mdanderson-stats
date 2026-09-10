"""Published trial-prefix timing, causal replay and simulated timing distributions."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import KeyboardDesign, run_tite_keyboard_trial, simulate_tite_keyboard


def test_staggered_safe_cohorts_wait_for_two_ascertained_patients():
    trial = run_tite_keyboard_trial(KeyboardDesign(), [15] * 6, np.full((6, 2), np.inf), 90)
    assert_array_equal(trial.enrollment_times, [15, 30, 45, 120, 135, 150])
    assert_array_equal(trial.assigned_doses, [1, 1, 1, 2, 2, 2])
    assert [s.time for s in trial.steps] == [60, 105, 120]
    assert [s.decision.action for s in trial.steps] == [
        "suspend_pending",
        "suspend_pending",
        "escalate",
    ]
    assert trial.suspension_time == 60
    assert trial.final_time == 240
    assert trial.selected_dose == 2
    scaled = run_tite_keyboard_trial(
        KeyboardDesign(), np.full(6, 15e-100), np.full((6, 2), np.inf), 90e-100
    )
    assert_allclose(scaled.enrollment_times / 1e-100, trial.enrollment_times)
    assert_allclose(scaled.final_time / 1e-100, trial.final_time)


def test_paper_delayed_dlt_prefix_and_no_future_outcome_leakage():
    potential = np.full((9, 2), np.inf)
    potential[3, 1] = 25  # Patient 4: enrolled on day 120, DLT observed on day 145.
    trial = run_tite_keyboard_trial(
        KeyboardDesign(), [15] * 9, potential, 90, pending_fraction_limit=None
    )
    assert_array_equal(trial.assigned_doses, [1, 1, 1, 2, 2, 2, 1, 1, 1])
    decision = next(s for s in trial.steps if s.time == 165)
    assert decision.decision.action == "deescalate"
    assert_allclose(decision.decision.effective_sample_size[1], 1.5)
    changed = potential.copy()
    changed[4, 1] = 40  # Still unobserved at day 165; must not affect that decision.
    future = run_tite_keyboard_trial(
        KeyboardDesign(), [15] * 9, changed, 90, pending_fraction_limit=None
    )
    assert_array_equal(future.assigned_doses, trial.assigned_doses)
    assert_allclose(
        next(s for s in future.steps if s.time == 165).decision.posterior.probability,
        decision.decision.posterior.probability,
    )
    assert future.toxicities[1] == trial.toxicities[1] + 1


def test_safety_and_precision_termination_finish_followup():
    unsafe = run_tite_keyboard_trial(KeyboardDesign(), [15] * 6, np.zeros((6, 2)), 90)
    assert unsafe.stop_reason == "stop_safety"
    assert unsafe.selected_dose is None
    assert_array_equal(unsafe.patients, [3, 0])
    precision = run_tite_keyboard_trial(
        KeyboardDesign(early_stop_patients=3), [15] * 6, np.full((6, 2), np.inf), 90
    )
    assert precision.stop_reason == "stop_enrollment"
    assert precision.final_time == 135
    assert precision.steps[-1].time == 120
    assert_array_equal(precision.patients, [3, 0])


def test_conditional_event_timing_and_arrival_generator():
    uniform = simulate_tite_keyboard(
        KeyboardDesign(), [1, 1], 90, 1 / 15, cohorts=1, cohort_size=1, trials=2000, rng=136
    )
    fractions = (uniform.duration - 15) / 90
    assert np.all((fractions >= 0) & (fractions <= 1))
    assert abs(np.mean(fractions > 0.5) - 0.5) < 6 * np.sqrt(0.25 / 2000)
    late = simulate_tite_keyboard(
        KeyboardDesign(),
        [1, 1],
        90,
        1 / 15,
        cohorts=1,
        cohort_size=1,
        trials=1000,
        rng=137,
        event_trimester_probabilities=[0, 0, 1],
    )
    assert np.all((late.duration >= 75) & (late.duration <= 105))
    arrivals = simulate_tite_keyboard(
        KeyboardDesign(),
        [1, 1],
        0.001,
        2,
        cohorts=1,
        cohort_size=1,
        trials=2000,
        rng=138,
        arrival="exponential",
    )
    assert abs(arrivals.duration.mean() - 0.5005) < 6 * 0.5 / np.sqrt(2000)
