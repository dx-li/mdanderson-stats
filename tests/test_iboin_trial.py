"""Deterministic accelerated-titration conduct traces."""

import numpy as np
import pytest

from mdanderson_stats.iboin import IBOINDesign
from mdanderson_stats.iboin_trial import replay_iboin_trial


def _design() -> IBOINDesign:
    return IBOINDesign([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0])


def test_lower_cap_without_toxicity_hands_off_to_full_cohort() -> None:
    result = replay_iboin_trial(
        _design(),
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        cohort_size=2,
        titration_cap=2,
        max_patients=5,
    )
    np.testing.assert_array_equal(result.assigned_dose, [1, 2, 3, 3, 4])
    np.testing.assert_array_equal(result.patients, [1, 1, 2, 1])
    assert result.titration_end_reason == "dose_cap"
    assert len(result.decision_history) == 1
    assert result.decision_history[0].current_dose == 3
    assert result.phase == "stopped"
    assert result.stop_reason == "stop_max_patients"
    assert result.next_dose is None
    assert result.cohort_remaining == 1


def test_dlt_tops_up_current_dose_then_uses_iboin_decision() -> None:
    result = replay_iboin_trial(
        _design(),
        [1, 0, 0],
        [0, 0, 0],
        cohort_size=2,
        titration_cap=1,
        max_patients=6,
    )
    np.testing.assert_array_equal(result.assigned_dose, [1, 1, 1])
    assert result.titration_end_reason == "DLT"
    assert result.decision_history[0].current_dose == 1
    assert result.decision_history[0].decision.next_dose == 1
    assert result.next_dose == 1
    assert result.phase == "cohort"
    assert result.cohort_remaining == 1


def test_second_grade2_event_triggers_top_up() -> None:
    result = replay_iboin_trial(
        _design(),
        [0, 0, 0],
        [0, 1, 1],
        cohort_size=2,
        max_patients=6,
    )
    np.testing.assert_array_equal(result.assigned_dose, [1, 2, 3])
    assert result.titration_end_reason == "grade2"
    assert result.phase == "cohort"
    assert result.next_dose == 3
    assert result.cohort_remaining == 1


def test_highest_dose_trigger_with_cohort_size_one_still_decides() -> None:
    result = replay_iboin_trial(
        _design(),
        [0],
        [0],
        cohort_size=1,
        starting_dose=4,
        max_patients=1,
    )
    np.testing.assert_array_equal(result.assigned_dose, [4])
    assert result.titration_end_reason == "highest_dose"
    assert len(result.decision_history) == 1
    assert result.decision_history[0].current_dose == 4
    assert result.stop_reason == "stop_max_patients"
    assert result.next_dose is None


def test_design_stop_rejects_unused_trailing_outcomes() -> None:
    with pytest.raises(ValueError, match="after the design stopped"):
        replay_iboin_trial(
            _design(),
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            cohort_size=3,
            max_patients=10,
        )
