import numpy as np
import pytest

from mdanderson_stats.boin_waterfall import BOINWaterfall, next_subtrial


def test_next_subtrial_matches_boin_waterfall_example():
    patients = np.array([[6, 0, 0, 0], [6, 0, 0, 0], [9, 12, 0, 0]])
    toxicities = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 3, 0, 0]])
    plan = next_subtrial(0.3, patients, toxicities)
    assert plan.candidate_mtd == (3, 2)
    assert plan.next_subtrial == ((2, 2), (2, 3), (2, 4))
    assert plan.starting_dose == (2, 3)
    assert plan.current_subtrial == 3


def test_r_keyword_aliases_and_terminal_first_row():
    patients = np.array([[6, 0, 0], [6, 10, 12]])
    toxicities = np.array([[0, 0, 0], [1, 1, 4]])
    plan = next_subtrial(target=0.3, npts=patients, ntox=toxicities)
    assert plan.next_subtrial is not None
    assert plan.starting_dose[0] == 1

    complete = next_subtrial(
        0.3,
        np.array([[0, 4, 3], [0, 0, 0]]),
        np.array([[0, 1, 1], [0, 0, 0]]),
    )
    assert complete.current_subtrial == 1
    assert complete.next_subtrial is None
    assert complete.action == "complete"


def test_safety_stop_and_validation():
    plan = BOINWaterfall(target=0.3).next_subtrial(
        np.array([[3, 0], [0, 0]]), np.array([[3, 0], [0, 0]])
    )
    assert plan.action == "stop_safety"
    assert plan.next_subtrial is None
    with pytest.raises(ValueError, match="cannot exceed"):
        next_subtrial(0.3, np.ones((2, 2)), np.array([[0, 2], [0, 0]]))
