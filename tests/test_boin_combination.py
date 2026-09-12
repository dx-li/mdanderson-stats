import numpy as np

from mdanderson_stats.boin_combination import BOINCombDesign


def test_public_example_selects_the_reported_single_mtd() -> None:
    patients = np.array([[6, 3, 0, 0], [6, 24, 9, 0], [0, 0, 0, 0]])
    toxicities = np.array([[0, 0, 0, 0], [1, 5, 4, 0], [0, 0, 0, 0]])

    result = BOINCombDesign(target=0.25).select_mtd(patients, toxicities)

    assert result.dose == (2, 2)
    assert np.isnan(result.isotonic_mean[0, 2])
    assert np.isfinite(result.isotonic_mean[1, 2])


def test_next_dose_uses_both_adjacent_moves_and_reproducible_rng() -> None:
    patients = np.zeros((3, 4), dtype=int)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 3
    design = BOINCombDesign(target=0.25)

    first = design.next_dose(patients, toxicities, (1, 1), rng=17)
    second = design.next_dose(patients, toxicities, (1, 1), rng=17)

    assert first.action == "escalate"
    assert first.next_dose == second.next_dose
    assert first.next_dose in {(1, 2), (2, 1)}


def test_unsafe_lowest_dose_stops_and_no_data_has_no_mtd() -> None:
    design = BOINCombDesign(target=0.3)
    patients = np.zeros((2, 3), dtype=int)
    toxicities = np.zeros_like(patients)
    assert design.select_mtd(patients, toxicities).dose is None

    patients[0, 0] = 3
    toxicities[0, 0] = 3
    decision = design.next_dose(patients, toxicities, (1, 1))
    assert decision.action == "stop_safety"
    assert decision.next_dose is None
    assert np.all(decision.eliminated)


def test_contour_returns_one_based_pairs_and_respects_monotonicity() -> None:
    patients = np.array([[6, 9, 0], [6, 9, 6]])
    toxicities = np.array([[0, 1, 0], [1, 3, 2]])

    result = BOINCombDesign(target=0.3).select_mtd(
        patients, toxicities, mtd_contour=True
    )

    assert result.dose is None
    assert result.contour
    assert all(1 <= a <= 2 and 1 <= b <= 3 for a, b in result.contour)
    assert all(left[1] <= right[1] for left, right in zip(result.contour, result.contour[1:]))
