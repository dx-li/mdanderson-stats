import numpy as np

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.boin12_simulation import simulate_boin12


def test_joint_multinomial_simulation_is_reproducible_and_conserves_counts() -> None:
    design = BOIN12Design(target_toxicity=0.3, toxicity_limit=0.35, efficacy_limit=0.25)
    grid = np.array(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.45, 0.20, 0.25, 0.10],
            [0.20, 0.15, 0.40, 0.25],
        ]
    )
    first = simulate_boin12(design, grid, cohorts=5, cohort_size=3, trials=24, rng=11)
    second = simulate_boin12(design, grid, cohorts=5, cohort_size=3, trials=24, rng=11)

    assert np.array_equal(first.patients, second.patients)
    assert np.array_equal(first.toxicities, second.toxicities)
    assert np.array_equal(first.efficacies, second.efficacies)
    assert np.array_equal(first.efficacy_without_toxicity, second.efficacy_without_toxicity)
    assert np.array_equal(first.selected_obd, second.selected_obd)
    assert np.all(first.toxicities <= first.patients)
    assert np.all(first.efficacies <= first.patients)
    assert np.all(first.efficacy_without_toxicity <= first.efficacies)
    assert np.all(first.efficacy_without_toxicity <= first.patients - first.toxicities)
    assert np.all(first.patients.sum(axis=1) <= 15)
    assert np.isclose(first.obd_probability.sum(), 1.0)
    assert np.isclose(first.mtd_probability.sum(), 1.0)


def test_joint_rows_are_validated_and_all_toxic_trials_stop_without_selection() -> None:
    design = BOIN12Design(target_toxicity=0.3, toxicity_limit=0.35, efficacy_limit=0.25)
    with np.testing.assert_raises(ValueError):
        simulate_boin12(design, [[0.5, 0.5, 0.0, 0.1]])

    result = simulate_boin12(
        design,
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        cohorts=4,
        cohort_size=3,
        trials=12,
        rng=3,
    )
    assert np.all(result.selected_obd == 0)
    assert np.all(result.selected_mtd == 0)
    assert set(result.stop_reason) == {"stop_no_admissible_neighbor"}


def test_inadmissible_upper_current_dose_deescalates_to_admissible_lower_dose() -> None:
    design = BOIN12Design(target_toxicity=0.3, toxicity_limit=0.35, efficacy_limit=0.25)
    patients = np.array([3, 3, 0])
    toxicities = np.array([0, 3, 0])
    efficacies = np.array([1, 1, 0])
    efficacy_without_toxicity = np.array([1, 0, 0])

    decision = design.next_dose(
        patients,
        toxicities,
        efficacies,
        2,
        efficacy_without_toxicity=efficacy_without_toxicity,
        eliminated=np.array([False, True, False]),
    )

    assert decision.action == "deescalate"
    assert decision.next_dose == 1
