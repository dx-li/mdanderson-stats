import pytest

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.boin import BOINDesign


def test_guide_one_of_three_modifier_is_target_specific_and_changes_action():
    design = BFBOINDesign(target=0.25, stay_at_one_of_three=True)
    assert design.next_dose([3, 0], [0, 0], [3, 0], 1).action == "escalate"
    assert design.next_dose([0, 3], [0, 1], [0, 3], 2).action == "stay"
    assert design.next_dose([0, 3], [0, 2], [0, 3], 2).action == "deescalate"
    with pytest.raises(ValueError, match="target == 0.25"):
        BFBOINDesign(target=0.26, stay_at_one_of_three=True)


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
    assert not design.next_dose([3, 0], [2, 0], [3, 0], 1).eliminated[0]
    assert design.next_dose([4, 0], [3, 0], [4, 0], 1).eliminated[0]
    table = design.boundary_table(4)
    assert table.lowest_stop_min[2] == table.eliminate_min[2]
    assert table.lowest_stop_min[3] <= 4

    strict = BFBOINDesign(target=0.25, bound_mtd=True)
    object.__setattr__(strict, "_boin", BOINDesign.from_boundaries(0.25, 0.1, 0.5))
    inclusive = BFBOINDesign(target=0.25, bound_mtd=False)
    object.__setattr__(inclusive, "_boin", BOINDesign.from_boundaries(0.25, 0.1, 0.5))
    assert strict.select_mtd([2, 0], [1, 0]).dose is None
    assert inclusive.select_mtd([2, 0], [1, 0]).dose == 1
