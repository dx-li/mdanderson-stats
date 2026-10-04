import pytest

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.boin import BOINDesign


def test_guide_one_of_three_modifier_is_target_specific_and_changes_action():
    design = BFBOINDesign(target=0.25, stay_at_one_of_three=True)
    assert design.next_dose([3, 0], [0, 0], [3, 0], 1).action == "escalate"
    assert design.next_dose([0, 3], [0, 1], [0, 3], 2).action == "stay"
    assert design.next_dose([0, 3], [0, 2], [0, 3], 2).action == "deescalate"
    for target in (0.20, 0.279):
        design = BFBOINDesign(
            target=target,
            stay_at_one_of_three=True,
            elimination_probability=0.999999,
        )
        for toxicities, expected in enumerate(("escalate", "stay", "deescalate", "deescalate")):
            assert design.next_dose([0, 3, 0], [0, toxicities, 0], [0, 3, 0], 2).action == expected
        table = design.boundary_table(3)
        assert table.deescalate_min[-1] == 2
    for target in (0.199, 0.28):
        with pytest.raises(ValueError, match=r"target in \[0.20,0.279\]"):
            BFBOINDesign(target=target, stay_at_one_of_three=True)


@pytest.mark.parametrize("target", [0.28, 0.33])
def test_guide_two_of_six_modifier_uses_all_source_count_categories(target):
    design = BFBOINDesign(
        target=target,
        deescalate_at_two_of_six=True,
        elimination_probability=0.999999,
    )
    for toxicities in range(7):
        expected = "escalate" if toxicities <= 1 else "deescalate"
        decision = design.next_dose([0, 6, 0], [0, toxicities, 0], [0, 6, 0], 2)
        assert decision.action == expected
    table = design.boundary_table(6)
    assert table.escalate_max[-1] == 1
    assert table.deescalate_min[-1] == 2
    with pytest.raises(ValueError, match=r"target in \[0.28,0.33\]"):
        BFBOINDesign(target=0.279, deescalate_at_two_of_six=True)


def test_unmodified_bf_boundaries_and_actions_retain_original_boin_behavior():
    for target, patients, toxicities, current in (
        (0.25, [3, 4, 0], [1, 2, 0], 2),
        (0.30, [6, 5, 2], [2, 2, 1], 2),
    ):
        bf = BFBOINDesign(target=target)
        ordinary = BOINDesign(target=target)
        assert (
            bf.next_dose(patients, toxicities, patients, current).action
            == ordinary.next_dose(patients, toxicities, current).action
        )
        bf_table, ordinary_table = bf.boundary_table(12), ordinary.boundary_table(12)
        for name in (
            "patients",
            "escalate_max",
            "deescalate_min",
            "eliminate_min",
            "lowest_stop_min",
        ):
            assert (getattr(bf_table, name) == getattr(ordinary_table, name)).all()


def test_bf_modifiers_do_not_change_ordinary_boin_and_safety_still_dominates():
    # BF-specific .20 target support does not relax the ordinary BOIN modifier.
    with pytest.raises(ValueError):
        BOINDesign(target=0.20, stay_at_one_of_three=True)
    design = BFBOINDesign(
        target=0.28,
        deescalate_at_two_of_six=True,
        elimination_probability=0.5,
    )
    decision = design.next_dose([6, 0], [6, 0], [6, 0], 1)
    assert decision.action == "stop_safety"
    assert decision.next_dose is None


def test_two_of_six_individual_action_enters_existing_backfill_conflict_rule():
    design = BFBOINDesign(target=0.30, deescalate_at_two_of_six=True)
    decision = design.next_dose(
        [3, 6],
        [2, 1],
        [3, 6],
        2,
        backfilled=[True, False],
    )
    # 1/6 would escalate; the lower backfilled 2/3 action conflicts. The
    # pooled 3/9 rate is between BOIN boundaries and therefore stays.
    assert decision.action == "stay"
    assert decision.next_dose == 2


def test_one_of_three_modifier_precedes_existing_backfill_conflict_pooling():
    design = BFBOINDesign(target=0.25, stay_at_one_of_three=True)
    decision = design.next_dose(
        [3, 3],
        [2, 1],
        [3, 3],
        2,
        backfilled=[True, False],
        response_observed=[True, True],
    )
    # The current-dose individual action is modified to stay, then the
    # conflicting lower-dose action is still resolved by pooled-rate rules.
    assert decision.action == "deescalate"
    assert decision.next_dose == 1


def test_bf_extra_safety_uses_strictly_more_than_three_and_strict_mtd_bound():
    design = BFBOINDesign(
        target=0.25, elimination_probability=0.99, extra_safe=True, safety_offset=0.05
    )
    three = design.next_dose([3, 0], [2, 0], [3, 0], 1)
    four = design.next_dose([4, 0], [3, 0], [4, 0], 1)
    # Independent exact Beta(3,2) and Beta(4,2) upper tails at one quarter.
    assert three.overdose_probability[0] == pytest.approx(243 / 256)
    assert four.overdose_probability[0] == pytest.approx(63 / 64)
    assert not three.eliminated[0]
    assert four.eliminated[0]
    assert design.select_mtd([3, 0], [2, 0]).dose == 1
    assert design.select_mtd([4, 0], [3, 0]).dose is None
    ordinary = BOINDesign(
        target=0.25, elimination_probability=0.99, extra_safe=True, safety_offset=0.05
    )
    assert ordinary.next_dose([3, 0], [2, 0], 1).eliminated[0]
    table = design.boundary_table(4)
    assert table.lowest_stop_min[2] == table.eliminate_min[2]
    assert table.lowest_stop_min[3] <= 4

    strict = BFBOINDesign(target=0.25, bound_mtd=True)
    object.__setattr__(strict, "_boin", BOINDesign.from_boundaries(0.25, 0.1, 0.5))
    inclusive = BFBOINDesign(target=0.25, bound_mtd=False)
    object.__setattr__(inclusive, "_boin", BOINDesign.from_boundaries(0.25, 0.1, 0.5))
    assert strict.select_mtd([2, 0], [1, 0]).dose is None
    assert inclusive.select_mtd([2, 0], [1, 0]).dose == 1
