"""KeyboardComb cohort simulation and operating-characteristic checks."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats.keyboard_combination import KeyboardCombDesign
from mdanderson_stats.keyboard_combination_simulation import simulate_keyboard_combination


def test_all_safe_scenario_completes_cohorts_and_selects_a_treated_pair():
    design = KeyboardCombDesign(early_stop_patients=100)
    result = simulate_keyboard_combination(
        design,
        np.zeros((2, 2)),
        cohorts=4,
        cohort_size=3,
        trials=12,
        rng=123,
    )
    assert_array_equal(result.toxicities, 0)
    assert_array_equal(result.eliminated, False)
    assert_array_equal(result.patients.sum(axis=(1, 2)), 12)
    assert np.all((result.selected_dose[:, 0] >= 1) & (result.selected_dose[:, 0] <= 2))
    assert np.all((result.selected_dose[:, 1] >= 1) & (result.selected_dose[:, 1] <= 2))
    assert result.stop_reason == ("max_cohorts",) * 12


def test_all_toxic_scenario_stops_at_the_lowest_pair_without_selection():
    result = simulate_keyboard_combination(
        KeyboardCombDesign(),
        np.ones((2, 2)),
        cohorts=6,
        cohort_size=3,
        trials=10,
        rng=5,
    )
    assert_array_equal(result.patients[:, 0, 0], 3)
    assert_array_equal(result.toxicities[:, 0, 0], 3)
    assert_array_equal(result.patients.sum(axis=(1, 2)), 3)
    assert result.stop_reason == ("stop_safety",) * 10
    assert_array_equal(result.selected_dose, 0)
    assert result.selection_probability[0, 0] == 1
    assert result.selection_mcse[0, 0] == 0


def test_seeded_random_ties_are_reproducible_and_have_nonzero_mcse():
    design = KeyboardCombDesign(early_stop_patients=24)
    truth = np.array([[0.10, 0.20, 0.30], [0.20, 0.30, 0.40], [0.30, 0.40, 0.50]])
    kwargs = dict(cohorts=8, cohort_size=3, trials=300, rng=421)
    first = simulate_keyboard_combination(design, truth, **kwargs)
    second = simulate_keyboard_combination(design, truth, **kwargs)
    assert_array_equal(first.patients, second.patients)
    assert_array_equal(first.toxicities, second.toxicities)
    assert_array_equal(first.selected_dose, second.selected_dose)
    selected = first.selection_probability > 0
    assert selected.sum() >= 2
    assert np.count_nonzero(first.selection_mcse[selected]) >= 2
    expected = np.sqrt(first.selection_probability * (1 - first.selection_probability) / 300)
    np.testing.assert_allclose(first.selection_mcse, expected)


@pytest.mark.parametrize("bad", [np.zeros((2, 1)), np.zeros((1, 2)), np.full((2, 2), -0.1)])
def test_truth_grid_validation(bad):
    with pytest.raises(ValueError):
        simulate_keyboard_combination(KeyboardCombDesign(), bad, trials=2)
