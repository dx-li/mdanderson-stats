"""Live app 2.5.4.0 rules for pending and complete 1/3 and 2/6 data."""

import numpy as np
from numpy.testing import assert_array_equal

from mdanderson_stats import BOINDesign, run_tite_boin_trial, tite_boin_decision, tite_boin_estimate


def test_one_of_three_requires_all_outcomes_known():
    design = BOINDesign(target=0.25, stay_at_one_of_three=True)
    result = tite_boin_estimate(design, 3, 1, [0, 1, 2], [0, 0.9, 1.8])
    assert_array_equal(result.move, [0, -1, -1])
    assert np.isneginf(result.deescalate_stft[0])
    for pending in range(3):
        step = tite_boin_decision(design, [0, 3], [0, 1], [[], [81] * pending], 2, 90)
        assert step.action == ("stay" if pending == 0 else "deescalate")
    potential = np.full((6, 2), np.inf)
    potential[0, 1] = 0
    replay = run_tite_boin_trial(design, [0, 0, 0, 100, 0, 0], potential, 90, start_dose=2)
    assert_array_equal(replay.assigned_doses, [2] * 6)


def test_two_of_six_overrides_pending_suspension():
    design = BOINDesign(target=0.3, deescalate_at_two_of_six=True)
    pending = np.arange(5)
    result = tite_boin_estimate(design, 6, 2, pending, pending * 0.99)
    assert_array_equal(result.move, [-1] * 5)
    assert np.all(np.isposinf(result.deescalate_stft))
    for c in pending:
        step = tite_boin_decision(design, [0, 6], [0, 2], [[], [89.1] * c], 2, 90)
        assert step.action == "deescalate"
        assert step.next_dose == 1
    ordinary = tite_boin_decision(BOINDesign(target=0.3), [0, 6], [0, 2], [[], [89.1] * 4], 2, 90)
    assert ordinary.action == "suspend_pending"


def test_complete_data_agrees_with_modified_boin_and_safety_still_wins():
    for design in (
        BOINDesign(target=0.25, stay_at_one_of_three=True),
        BOINDesign(target=0.3, deescalate_at_two_of_six=True),
    ):
        for n in range(1, 10):
            for y in range(n + 1):
                ordinary = design.next_dose([0, n, 0], [0, y, 0], 2)
                tite = tite_boin_decision(design, [0, n, 0], [0, y, 0], [[], [], []], 2, 90)
                assert tite.action == ordinary.action
                assert tite.next_dose == ordinary.next_dose
        stopped = tite_boin_decision(design, [3, 6], [3, 2], [[], [80] * 4], 2, 90)
        assert stopped.action == "stop_safety"
