import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.bard_blrm_decision import (
    bard_blrm_backfill,
    bard_blrm_next_dose,
    bard_blrm_select_mtd,
)


def test_independent_hand_calculated_decision_snapshots() -> None:
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "bard-blrm-decisions.json").read_text()
    )
    functions = {
        "next_dose": bard_blrm_next_dose,
        "backfill": bard_blrm_backfill,
        "select_mtd": bard_blrm_select_mtd,
    }
    for case in cases:
        result = functions[case["function"]](**case["inputs"])
        for name, expected in case["expected"].items():
            actual = getattr(result, name)
            if isinstance(actual, np.ndarray):
                actual = actual.tolist()
            elif isinstance(actual, tuple):
                actual = list(actual)
            assert actual == expected, (case["case"], name, actual, expected)


def test_next_dose_uses_strict_safety_one_step_and_stable_ties() -> None:
    # The safe PTT tie chooses the lower dose; movement remains one level.
    # A separate case exposes an unsafe intermediate rather than skipping it.
    move = bard_blrm_next_dose([0.5, 0.5, 0.4], [0.1, 0.1, 0.6], 3, eta=0.5)
    assert move.action == "deescalate"
    assert move.target_dose == 1
    assert move.tied_target_doses == (1, 2)
    assert move.next_dose == 2
    assert move.next_dose_safe is True

    unsafe_step = bard_blrm_next_dose([0.5, 0.69, 0.6], [0.1, 0.31, 0.4], 3, eta=0.3)
    assert unsafe_step.target_dose == 1
    assert unsafe_step.next_dose == 2
    assert unsafe_step.next_dose_safe is False
    assert not move.safe.flags.writeable

    equality = bard_blrm_next_dose([0.7, 0.2], [0.3, 0.4], 1, eta=0.3)
    assert equality.action == "no_eligible_safe_dose"
    assert equality.next_dose is None
    all_over = bard_blrm_next_dose([0.69, 0.2], [0.31, 0.4], 1, eta=0.3)
    assert all_over.action == "stop_all_overdose"


def test_backfill_uses_response_at_or_below_and_evaluable_cap() -> None:
    result = bard_blrm_backfill(
        [0.1, 0.2, 0.3, 0.4],
        responses=[0, 1, 1, 1],
        evaluable=[2, 3, 1, 6],
        assigned=[2, 4, 3, 6],
        current_dose=4,
        eta=0.3,
        cap=6,
    )
    assert result.eligible.tolist() == [False, True, False, False]
    assert result.closed.tolist() == [False, False, True, True]
    assert result.selected_dose == 2

    no_response = bard_blrm_backfill([0.1, 0.2, 0.25], [0, 0, 0], [0, 0, 0], 3, eta=0.3, cap=6)
    assert no_response.selected_dose is None


def test_final_selection_uses_treated_count_and_safe_ptt_tie_rule() -> None:
    result = bard_blrm_select_mtd([0.9, 0.7, 0.69], [0.1, 0.2, 0.31], [5, 6, 20], eta=0.3)
    assert result.selected_dose == 2
    assert result.eligible.tolist() == [False, True, False]
    assert result.status == "selected"

    unsafe = bard_blrm_select_mtd([0.7, 0.5], [0.3, 0.4], [8, 8], eta=0.3)
    assert unsafe.selected_dose is None
    assert unsafe.status == "no_eligible_safe_dose"


def test_probability_contract_rejects_incoherent_or_nonmonotone_pod() -> None:
    with pytest.raises(ValueError, match=r"ptt \+ pod"):
        bard_blrm_next_dose([0.8], [0.3], 1)
    with pytest.raises(ValueError, match="nondecreasing"):
        bard_blrm_next_dose([0.2, 0.3], [0.4, 0.2], 1)
    with pytest.raises(ValueError, match="responses and evaluable"):
        bard_blrm_backfill([0.1], [2], [1], 1, cap=4)
    with pytest.raises(ValueError, match="pod must be real-valued"):
        bard_blrm_backfill([0.1 + 0.2j], [0], [0], 1, cap=4)
