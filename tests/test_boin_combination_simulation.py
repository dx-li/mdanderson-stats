import numpy as np
import pytest

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


def test_precision_stop_allows_escalation_until_the_highest_combination() -> None:
    design = BOINCombDesign(early_stop_patients=3)
    result = simulate_boin_combination(
        design, np.zeros((2, 3)), cohorts=8, cohort_size=3, trials=8, rng=128
    )
    # All-safe trials require three escalations before reaching the highest
    # combination. The enrollment threshold must not stop the first cohort.
    assert np.all(result.patients.sum(axis=(1, 2)) == 12)
    assert np.all(result.patients[:, 1, 2] == 3)
    assert set(result.stop_reason) == {"stop_precision"}
    toxic = simulate_boin_combination(
        design, np.ones((2, 3)), cohorts=8, cohort_size=3, trials=8, rng=128
    )
    assert set(toxic.stop_reason) == {"stop_safety"}


def test_accelerated_titration_staircase_and_first_cohort_top_up() -> None:
    result = simulate_boin_combination(
        BOINCombDesign(target=0.3, early_stop_patients=None),
        np.zeros((2, 2)),
        cohorts=1,
        cohort_size=3,
        trials=4,
        titration=True,
        rng=91,
    )

    assert np.all(result.titration_patients.sum(axis=(1, 2)) == 3)
    assert np.all(result.titration_patients[:, 0, 0] == 1)
    assert np.all(result.titration_patients[:, 1, 1] == 1)
    assert np.all(result.titration_endpoint == (2, 2))
    assert set(result.titration_end_reason) == {"upper_right"}
    # Three one-patient staircase cells are followed by a two-patient top-up.
    assert np.all(result.patients.sum(axis=(1, 2)) == 5)
    assert np.all(result.patients[:, 1, 1] == 3)
    assert np.array_equal(result.toxicities, np.zeros_like(result.toxicities))


def test_accelerated_titration_first_dlt_hands_off_at_current_cell() -> None:
    result = simulate_boin_combination(
        BOINCombDesign(target=0.3, early_stop_patients=None),
        np.ones((2, 2)),
        cohorts=2,
        cohort_size=3,
        trials=3,
        titration=True,
        rng=22,
    )

    assert np.all(result.titration_patients[:, 0, 0] == 1)
    assert np.all(result.titration_patients.sum(axis=(1, 2)) == 1)
    assert np.all(result.titration_endpoint == (1, 1))
    assert set(result.titration_end_reason) == {"first_dlt"}
    assert np.all(result.patients[:, 0, 0] == 3)
    assert np.all(result.toxicities[:, 0, 0] == 3)
    assert np.all(result.patients.sum(axis=(1, 2)) == 3)
    assert set(result.stop_reason) == {"stop_safety"}


def test_cohort_size_one_disables_titration_and_budget_includes_staircase() -> None:
    disabled = simulate_boin_combination(
        BOINCombDesign(target=0.3),
        np.zeros((2, 2)),
        cohorts=1,
        cohort_size=1,
        trials=1,
        titration=True,
        rng=9,
    )
    assert disabled.titration_end_reason == ("cohort_size_one",)
    assert disabled.titration_patients.sum() == 0
    with pytest.raises(ValueError, match="1000 patients"):
        simulate_boin_combination(
            BOINCombDesign(target=0.3),
            np.zeros((2, 3)),
            cohorts=333,
            cohort_size=3,
            trials=1,
            titration=True,
            rng=9,
        )


def test_storage_preflight_rejects_before_advancing_the_supplied_rng() -> None:
    generator = np.random.default_rng(45)
    expected = np.random.default_rng(45).random()
    with pytest.raises(ValueError, match="128 MiB"):
        simulate_boin_combination(
            BOINCombDesign(target=0.3),
            np.zeros((2, 2)),
            cohorts=1,
            cohort_size=1,
            trials=500_000,
            rng=generator,
        )
    assert generator.random() == expected
