"""Focused BF-BOIN decision checks using small hand-computed count vectors."""

import numpy as np
import pytest

from mdanderson_stats.bf_boin import BFBOINDesign


def test_backfill_uses_assigned_cap_and_reopens_temporary_closure():
    design = BFBOINDesign(n_cap=12)
    open_state = design.backfill_eligibility(
        [3, 3, 0], [0, 0, 0], [4, 5, 0], 3, response_observed=[True, True, False]
    )
    assert open_state.dose == 2
    assert not open_state.eligible[1]
    reopened = design.backfill_eligibility(
        [3, 3, 3], [1, 1, 0], [4, 5, 3], 3, response_observed=[True, True, False]
    )
    assert reopened.closed[0]
    later = design.backfill_eligibility(
        [3, 6, 3], [1, 1, 0], [4, 8, 3], 3, response_observed=[True, True, False]
    )
    assert not later.closed[0]


def test_unobserved_activity_does_not_make_a_lower_dose_eligible():
    design = BFBOINDesign()
    result = design.backfill_eligibility(
        [0, 3, 0],
        [0, 0, 0],
        [3, 3, 0],
        3,
        response_observed=[False, False, False],
    )
    assert not np.any(result.eligible)


def test_pooled_conflict_escalates_when_backfill_pool_is_safe():
    design = BFBOINDesign()
    result = design.next_dose(
        [3, 3, 3],
        [0, 1, 0],
        [3, 3, 3],
        3,
        backfilled=[False, True, False],
        response_observed=[True, True, True],
    )
    assert result.action == "stay"
    assert result.next_dose == 3


def test_precision_stop_requires_assigned_current_count_and_stay():
    design = BFBOINDesign(n_stop=5)
    result = design.next_dose([3, 5], [0, 1], [4, 5], 2)
    assert result.action == "stop_precision"
    moving = design.next_dose([3, 5, 0], [0, 0, 0], [5, 5, 0], 2)
    assert moving.action == "escalate"


def test_invalid_fractional_current_dose_is_rejected():
    with pytest.raises(ValueError):
        BFBOINDesign().next_dose([3, 0], [0, 0], [3, 0], 1.5)
