import numpy as np

from mdanderson_stats.boin_combination import BOINCombDesign
from mdanderson_stats.boin_combination_simulation import simulate_boin_combination


def test_simulation_reproducibility_and_conservation() -> None:
    design = BOINCombDesign(target=0.3)
    true_toxicity = np.array([[0.05, 0.15, 0.3], [0.15, 0.3, 0.55]])
    first = simulate_boin_combination(
        design, true_toxicity, cohorts=5, cohort_size=3, trials=30, rng=123
    )
    second = simulate_boin_combination(
        design, true_toxicity, cohorts=5, cohort_size=3, trials=30, rng=123
    )

    assert np.array_equal(first.patients, second.patients)
    assert np.array_equal(first.toxicities, second.toxicities)
    assert np.array_equal(first.selected_dose, second.selected_dose)
    assert np.all(first.toxicities <= first.patients)
    assert np.all(first.patients.sum(axis=(1, 2)) <= 15)
    assert np.isclose(first.selection_probability.sum(), 1.0)


def test_simulation_stops_all_toxic_lowest_dose() -> None:
    result = simulate_boin_combination(
        BOINCombDesign(target=0.3),
        np.full((2, 2), 1.0),
        cohorts=4,
        cohort_size=3,
        trials=12,
        rng=7,
    )

    assert np.all(result.selected_dose == 0)
    assert set(result.stop_reason) == {"stop_safety"}
