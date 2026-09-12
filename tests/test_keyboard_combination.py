"""Numerical and safety checks for the two-dimensional Keyboard core."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betaincc

from mdanderson_stats.keyboard_combination import KeyboardCombDesign, _biviso


def test_weighted_bivariate_isotonic_regression_matches_manual_pooled_blocks():
    values = np.array([[0.8, 0.2], [0.1, 0.9]])
    weights = np.array([[2.0, 1.0], [3.0, 4.0]])
    result = _biviso(values, weights)
    assert_allclose(result, [[0.35, 0.35], [0.35, 0.9]], rtol=0, atol=2e-12)
    assert np.all(result[:, :-1] <= result[:, 1:])
    assert np.all(result[:-1, :] <= result[1:, :])


def test_bivariate_isotonic_regression_matches_iso_reference_with_unequal_weights():
    values = np.array(
        [[0.8, 0.1, 0.2, 0.9], [0.3, 0.7, 0.4, 0.6], [0.5, 0.2, 0.8, 0.9], [0.1, 0.4, 0.6, 0.3]]
    )
    weights = np.array(
        [[2, 1, 3, 4], [1.5, 2.2, 4.4, 3.3], [5, 1, 2, 6], [3, 7, 2.5, 1.2]]
    )
    expected = np.array(
        [
            [0.35806452, 0.35806451, 0.35806451, 0.76438356],
            [0.35806452, 0.43150685, 0.43150685, 0.76438356],
            [0.35806452, 0.43150685, 0.68888889, 0.8],
            [0.35806452, 0.43150685, 0.68888889, 0.8],
        ]
    )
    assert_allclose(_biviso(values, weights), expected, rtol=0, atol=5e-8)


def test_keyboard_comb_key_scores_use_stable_beta_tails_and_target_key():
    design = KeyboardCombDesign(target=0.3)
    assert design.target_key == 3
    assert_allclose(design.intervals[design.target_key], [0.25, 0.35])
    scores = design._key_scores(200, 0)
    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) < design.target_key
    assert design._move(3, 0) == 1
    assert design._move(200, 200) == -1


def test_next_dose_uses_southeast_safety_closure_and_samples_neighbor_ties():
    design = KeyboardCombDesign(target=0.3, early_stop_patients=None)
    patients = np.zeros((2, 3), dtype=int)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 3
    toxicities[0, 0] = 3
    step = design.next_dose(patients, toxicities, (1, 1), rng=42)
    assert step.action == "stop_safety"
    assert step.next_dose is None
    assert_array_equal(step.eliminated, [[True, True, True], [True, True, True]])
    # A safe, underdosed current cell has two equally informed untried neighbors;
    # the seeded generator must choose one of those two, never a diagonal jump.
    patients[:] = 0
    toxicities[:] = 0
    patients[0, 0] = 3
    step = design.next_dose(patients, toxicities, (1, 1), rng=7)
    assert step.action == "escalate"
    assert step.next_dose in {(1, 2), (2, 1)}


def test_final_selection_uses_cross_safety_closure_and_weighted_isotonic_fit():
    design = KeyboardCombDesign(target=0.3, early_stop_patients=None)
    patients = np.array([[0, 3, 0], [0, 3, 3]])
    toxicities = np.array([[0, 3, 0], [0, 0, 0]])
    result = design.select_mtd(patients, toxicities)
    # select.mtd.comb.kb's cross closure marks (1,1)'s row and column only.
    assert_array_equal(result.eliminated, [[False, True, True], [False, True, False]])
    assert result.dose == (2, 3)
    assert np.isfinite(result.isotonic_mean[0, 1])
    assert np.isfinite(result.isotonic_mean[1, 2])
    empty = design.select_mtd(np.zeros((2, 3)), np.zeros((2, 3)))
    assert empty.dose == (1, 1)
    assert np.isnan(empty.isotonic_mean).all()


def test_safety_posterior_matches_beta_binomial_tail():
    design = KeyboardCombDesign(target=0.3, early_stop_patients=None)
    n = np.array([[6, 0], [0, 0]])
    y = np.array([[2, 0], [0, 0]])
    result = design.next_dose(n, y, (1, 1))
    assert_allclose(result.overdose_probability[0, 0], betaincc(3, 5, 0.3), rtol=0, atol=2e-15)


def test_low_target_safety_allows_one_dlt_when_three_patients_are_evaluable():
    design = KeyboardCombDesign(target=0.05, early_stop_patients=None)
    assert design._cutoffs(3)[2] == 1
    patients = np.full((2, 2), 0)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 3
    toxicities[0, 0] = 1
    step = design.next_dose(patients, toxicities, (1, 1))
    assert step.action == "stop_safety"
    assert step.eliminated.all()


def test_eliminated_current_dose_never_falls_back_to_reenrollment():
    design = KeyboardCombDesign(early_stop_patients=None)
    patients = np.array([[3, 3], [3, 3]])
    toxicities = np.array([[0, 3], [3, 3]])
    step = design.next_dose(patients, toxicities, (2, 2))
    assert step.action == "stop_safety"
    assert step.next_dose is None
