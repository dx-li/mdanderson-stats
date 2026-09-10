import numpy as np

from mdanderson_stats import RareDisease123Design


def test_all_default_native_table_cells_with_explicit_uniform_prior():
    design = RareDisease123Design((1, 1))
    # Native E*/S chooses E when the next dose is untried; D/S chooses D
    # when the lower dose is admissible. S at a full dose terminates selection.
    expected = {
        1: ["ES", "DD"],
        3: ["EESS", "ESSS", "DDDD", "DDDD"],
        6: ["EEEESSS", "EEEESSS", "EEESSSS"] + ["DDDDDDD"] * 4,
    }
    for n, rows in expected.items():
        for toxicity, row in enumerate(rows):
            for response, action in enumerate(row):
                result = design.next_dose([0, n, 0], [0, toxicity, 0], [0, response, 0], 2)
                actual = {"escalate": "E", "stay": "S", "select_obd": "S", "deescalate": "D"}[
                    result.action
                ]
                assert actual == action, (n, toxicity, response)
    np.testing.assert_allclose(design.toxicity_boundaries, [0.221, 0.334], atol=0.0005)


def test_cohort_progression_and_selection_only_at_next_assigned_full_dose():
    design = RareDisease123Design((1, 1))
    for n, size in [(1, 2), (3, 3)]:
        step = design.next_dose([n, 0], [0, 0], [n, 0], 1)
        assert step.next_dose == 1 and step.cohort_size == size
        assert step.selected_dose is None
    selected = design.next_dose([6, 0], [0, 0], [6, 0], 1)
    assert selected.selected_dose == 1 and selected.next_dose is None
    # Even after the current dose is full, a different dose may need another cohort.
    move = design.next_dose([1, 6, 0], [0, 3, 0], [1, 6, 0], 2)
    assert move.next_dose == 1 and move.cohort_size == 2
    assert move.selected_dose is None


def test_exemptions_exclusions_and_exploration_limit():
    design = RareDisease123Design((1, 1))
    single = design.next_dose([1, 0, 0], [1, 0, 0], [0, 0, 0], 1)
    assert single.action == "stay" and single.cohort_size == 2
    assert single.admissible.all()
    toxic = design.next_dose([3, 0, 0], [2, 0, 0], [3, 0, 0], 1)
    assert toxic.action == "stop_no_obd"
    assert toxic.eliminated.all()
    futile = design.next_dose([3, 0, 0], [0, 0, 0], [0, 0, 0], 1)
    assert futile.next_dose == 2 and futile.cohort_size == 1
    np.testing.assert_array_equal(futile.eliminated, [True, False, False])
    # A previously well-explored higher dose is not revisited by escalation.
    stay = design.next_dose([3, 3, 0], [0, 0, 0], [1, 3, 0], 1)
    assert stay.next_dose == 1
    stop = design.next_dose([3, 1, 0], [0, 1, 0], [0, 1, 0], 2, eliminated=[True, True, False])
    assert stop.action == "stop_no_obd"
    assert not stay.admissible.flags.writeable
