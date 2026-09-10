"""Published Rolling Six decision states, terminal interpretation, and calendar paths."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import RollingSixDesign, run_rolling_six_trial, simulate_rolling_six


def test_all_feasible_single_dose_states_against_published_table():
    design = RollingSixDesign()
    for n in range(7):
        for y in range(n + 1):
            for c in range(n - y + 1):
                decision = design.next_dose([0, n, 0], [0, y, 0], [0, c, 0], 2)
                # PBTC-042 section 13.2 and the ordinary Rolling 6 column of
                # Frankel et al. Table 2. The n=0,1 states initialize enrollment.
                if y >= 2:
                    expected = "deescalate"
                elif (n, y, c) in {
                    (3, 0, 0),
                    (4, 0, 0),
                    (5, 0, 0),
                    (6, 0, 0),
                    (6, 0, 1),
                    (6, 1, 0),
                }:
                    expected = "escalate"
                elif n == 6:
                    expected = "suspend_pending"
                else:
                    expected = "stay"
                assert decision.action == expected, (n, y, c)
    with pytest.raises(ValueError, match="destination is full"):
        design.next_dose([3, 6, 0], [0, 0, 0], [0, 0, 0], 1)
    strict = RollingSixDesign(require_complete_before_escalation=True)
    assert strict.next_dose([6, 0], [0, 0], [1, 0], 1).action == "suspend_pending"


def test_downward_completion_and_highest_dose_interpretation():
    design = RollingSixDesign()
    lowered = design.next_dose([3, 2], [0, 2], [0, 0], 2)
    assert lowered.next_dose == 1
    assert lowered.action == "deescalate"
    waiting = design.next_dose([6, 2], [0, 2], [1, 0], 2)
    assert waiting.action == "suspend_pending"
    done = design.next_dose([6, 2], [1, 2], [0, 0], 2)
    assert done.action == "select_mtd"
    assert done.selected_dose == 1
    assert design.select_mtd([6, 2], [1, 2]).status == "mtd"
    assert design.select_mtd([3, 6], [0, 1]).status == "highest_planned_dose"
    assert design.select_mtd([3, 4], [0, 0]).status == "inconclusive"
    assert design.next_dose([2, 0], [2, 0], [0, 0], 1).action == "stop_safety"


def test_fast_accrual_resumes_with_five_known_non_dlts():
    potential = np.full((12, 2), np.inf)
    trial = run_rolling_six_trial(RollingSixDesign(), [1] * 12, potential, 10)
    assert_array_equal(trial.enrollment_times, [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20])
    assert_array_equal(trial.patients, [6, 6])
    assert trial.suspension_time == 8
    assert trial.final_time == 30
    assert trial.selection_status == "highest_planned_dose"
    strict = run_rolling_six_trial(RollingSixDesign(True), [1] * 12, potential, 10)
    assert strict.enrollment_times[6] == 16
    assert strict.final_time == 31
    toxic = run_rolling_six_trial(RollingSixDesign(), [1] * 12, np.zeros((12, 2)), 10)
    assert_array_equal(toxic.patients, [2, 0])
    assert toxic.selection_status == "no_safe_dose"


def test_simulation_extremes_and_dose_capacity():
    safe = simulate_rolling_six(RollingSixDesign(), [0, 0], 10, 1, trials=20, rng=6)
    assert_array_equal(safe.selection_probability, [0, 0, 1])
    assert set(safe.selection_status) == {"highest_planned_dose"}
    mixed = simulate_rolling_six(RollingSixDesign(), [0.1, 0.3, 0.5], 3, 2, trials=100, rng=6)
    assert np.all(mixed.patients <= 6)
    assert np.all(mixed.toxicities <= mixed.patients)
    for n, y, selected in zip(mixed.patients, mixed.toxicities, mixed.selected_dose):
        if selected:
            assert n[selected - 1] == 6
            assert y[selected - 1] <= 1
