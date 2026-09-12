import numpy as np
import pytest

from mdanderson_stats.uboin_conduct import UBOINDesign


def design(**kwargs: object) -> UBOINDesign:
    kwargs.setdefault("candidate_scope", "tried")
    return UBOINDesign(
        prior=np.full((2, 2), 0.25),
        utilities=np.array([[0, 30], [50, 100]]),
        **kwargs,
    )


def test_empty_start_and_stage_one_boundaries() -> None:
    d = design(s1=12, s2=24)
    empty = np.zeros((3, 2, 2), dtype=int)
    start = d.decision(empty, current_dose=1)
    assert start.action == "start"
    assert start.next_dose == 1
    safe = empty.copy()
    safe[0, 0, 0] = 3
    assert d.decision(safe, current_dose=1).action == "escalate"
    toxic = empty.copy()
    toxic[1, 0, 1] = 2
    toxic[1, 0, 0] = 1
    assert d.decision(toxic, current_dose=2).action == "deescalate"


def test_stage_one_safety_uses_fixed_point95_and_suffix() -> None:
    d = design(safety_cutoff=0.999)
    counts = np.zeros((3, 2, 2), dtype=int)
    counts[0, 0, 1] = 3
    result = d.decision(counts, current_dose=1)
    assert result.action == "stop_safety"
    assert np.array_equal(result.eliminated, [True, True, True])


def test_run_in_3plus3_and_target_validation() -> None:
    d = design(run_in_3plus3=True)
    one_of_three = np.zeros((3, 2, 2), dtype=int)
    one_of_three[0, 0, 1] = 1
    one_of_three[0, 0, 0] = 2
    assert d.decision(one_of_three, current_dose=1).action == "stay"
    two_of_six = np.zeros((3, 2, 2), dtype=int)
    two_of_six[1, 0, 1] = 2
    two_of_six[1, 0, 0] = 4
    assert d.decision(two_of_six, current_dose=2).action == "deescalate"
    with pytest.raises(ValueError):
        design(run_in_3plus3=True, delta=0.04)


def test_stage_two_b1_exploration_precedes_admissibility() -> None:
    d = design(s1=3)
    counts = np.zeros((3, 2, 2), dtype=int)
    counts[0, 0, 0] = 12
    result = d.decision(counts, current_dose=1, stage=1)
    assert result.stage == 2
    assert result.action == "escalate"
    assert result.next_dose == 2


def test_candidate_scope_and_final_selection() -> None:
    counts = np.zeros((3, 2, 2), dtype=int)
    counts[0, 1, 0] = 5
    tried = design(candidate_scope="tried").select_obd(counts)
    all_doses = design(candidate_scope="all").select_obd(counts)
    assert tried.dose == 1
    assert all_doses.dose in (1, 2, 3)
    assert not tried.eligible[1]
    assert all_doses.eligible[1]


def test_stage_two_allocation_and_caps() -> None:
    counts = np.zeros((2, 2, 2), dtype=int)
    counts[0, 0, 1] = 2
    counts[0, 1, 0] = 2
    d = design(s1=2, s2=20, method="equal")
    result = d.decision(counts, current_dose=1, stage=2)
    assert result.action == "assign"
    assert result.next_dose == 1
    assert result.allocation_probabilities[0] == 1
    capped = design(max_patients=4, s1=2, s2=20).decision(counts, current_dose=1, stage=2)
    assert capped.action == "stop_max_patients"
    assert capped.selected_dose == 1
    assert np.count_nonzero(capped.allocation_probabilities) == 0


def test_invalid_stage_two_empty_and_sticky_elimination() -> None:
    d = design()
    with pytest.raises(ValueError):
        d.decision(np.zeros((2, 2, 2), dtype=int), current_dose=1, stage=2)
    counts = np.zeros((2, 2, 2), dtype=int)
    counts[0, 1, 0] = 3
    result = d.decision(counts, current_dose=1, stage=2, eliminated=[False, True])
    assert result.action in ("assign", "stop_no_admissible")
    assert result.eliminated[1]
