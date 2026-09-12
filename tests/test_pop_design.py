import numpy as np

from mdanderson_stats.pop_design import PoPDesign, predictive_bayes_factor
from mdanderson_stats.pop_simulation import simulate_pop


def test_prbf_is_low_in_both_tails_and_one_at_target() -> None:
    values = predictive_bayes_factor(0.25, 10, np.arange(11))
    assert values[0] < values[3]
    assert values[-1] < values[3]
    assert values[3] < np.exp(1.0)


def test_paper_table_one_boundaries_target_quarter_thirty_patients() -> None:
    table = PoPDesign().boundaries(30, cohort_size=3)
    assert np.array_equal(table.patients, np.arange(3, 31, 3))
    assert np.array_equal(table.escalate_max, [0, 0, 1, 2, 2, 3, 4, 4, 5, 6])
    assert np.array_equal(table.deescalate_min, [2, 3, 3, 4, 5, 6, 7, 7, 8, 9])
    assert np.array_equal(table.exclude_under_max, [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2])
    assert np.array_equal(table.exclude_over_min, [3, 5, 6, 7, 8, 9, 11, 12, 13, 14])


def test_selection_handles_untried_doses_and_safety_filter() -> None:
    result = PoPDesign().select_mtd([3, 0, 3], [0, 0, 2])
    assert result.dose == 1
    assert np.isnan(result.isotonic_estimate[1])
    assert not result.eligible[1]


def test_simulation_is_reproducible_and_all_zero_escalates() -> None:
    design = PoPDesign()
    first = simulate_pop(design, np.zeros(3), total_patients=6, cohort_size=3, trials=4, seed=7)
    second = simulate_pop(design, np.zeros(3), total_patients=6, cohort_size=3, trials=4, seed=7)
    assert np.array_equal(first.selections, second.selections)
    assert np.all(first.patients[:, -1] == 4)


def test_all_toxic_stops_without_selection() -> None:
    result = simulate_pop(
        PoPDesign(), np.ones(2), total_patients=6, cohort_size=3, trials=3, seed=2
    )
    assert np.all(result.early_stop)
    assert np.all(result.selections == 0)
