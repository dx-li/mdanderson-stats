"""Compact checks for bCRM's explicit one-outcome decision rules."""

import pytest

from mdanderson_stats.bcrm_decision import bcrm_decision


def test_target_modes_equality_ties_and_unattainable_fallbacks():
    estimates = [0.1, 0.3, 0.5]
    assert (
        bcrm_decision(
            estimates, [0, 0, 0], target=0.3, max_subjects=12, selection="below"
        ).target_index
        == 1
    )
    assert (
        bcrm_decision(
            estimates, [0, 0, 0], target=0.3, max_subjects=12, selection="above"
        ).target_index
        == 1
    )
    # Equidistant nearest choices resolve toward the lower dose.
    assert (
        bcrm_decision(
            estimates, [0, 0, 0], target=0.4, max_subjects=12, selection="nearest"
        ).target_index
        == 1
    )
    assert (
        bcrm_decision(
            [0.1, 0.3], [0, 0], target=0.2, max_subjects=12, selection="nearest"
        ).target_index
        == 0
    )
    below = bcrm_decision([0.4, 0.5], [0, 0], target=0.3, max_subjects=12)
    above = bcrm_decision([0.1, 0.2], [0, 0], target=0.8, max_subjects=12, selection="above")
    assert (below.target_index, below.target_attainable) == (0, False)
    assert (above.target_index, above.target_attainable) == (1, False)


def test_start_dose_and_escalation_cap_use_highest_ever_tried():
    initial = bcrm_decision(
        [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], target=0.4, max_subjects=12, start_index=2
    )
    assert (initial.target_index, initial.next_index) == (3, 2)
    assert initial.target_attainable
    empty_fallback = bcrm_decision([0.4, 0.5], [0, 0], target=0.3, max_subjects=12, start_index=1)
    assert (empty_fallback.target_index, empty_fallback.next_index) == (0, 1)
    assert not empty_fallback.target_attainable
    capped = bcrm_decision(
        [0.1, 0.2, 0.3, 0.4], [0, 2, 0, 0], target=0.4, max_subjects=12, selection="nearest"
    )
    assert (capped.target_index, capped.next_index) == (3, 2)


def test_minimum_target_limit_and_maximum_stopping():
    estimates = [0.1, 0.2, 0.3]
    ongoing = bcrm_decision(estimates, [0, 6, 0], target=0.2, max_subjects=12, min_subjects=8)
    assert (ongoing.stop, ongoing.reason) == (False, "continue")
    target_stop = bcrm_decision(estimates, [0, 6, 0], target=0.2, max_subjects=12, min_subjects=6)
    assert (target_stop.stop, target_stop.next_index, target_stop.reason) == (
        True,
        None,
        "target_dose_limit",
    )
    max_stop = bcrm_decision(estimates, [6, 6, 0], target=0.2, max_subjects=12)
    assert (max_stop.stop, max_stop.next_index, max_stop.reason) == (True, None, "max_subjects")


def test_invalid_nonmonotone_probability_and_noncohort_history():
    with pytest.raises(ValueError, match="nondecreasing"):
        bcrm_decision([0.2, 0.1], [0, 0], target=0.2, max_subjects=12)
    with pytest.raises(ValueError, match="cohort multiple"):
        bcrm_decision([0.1, 0.2], [1, 0], target=0.2, max_subjects=12)
