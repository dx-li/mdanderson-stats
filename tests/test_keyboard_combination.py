"""Numerical and safety checks for the two-dimensional Keyboard core."""

import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betainc, betaincc

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
    weights = np.array([[2, 1, 3, 4], [1.5, 2.2, 4.4, 3.3], [5, 1, 2, 6], [3, 7, 2.5, 1.2]])
    expected = np.array(
        [
            [0.35806452, 0.35806451, 0.35806451, 0.76438356],
            [0.35806452, 0.43150685, 0.43150685, 0.76438356],
            [0.35806452, 0.43150685, 0.68888889, 0.8],
            [0.35806452, 0.43150685, 0.68888889, 0.8],
        ]
    )
    assert_allclose(_biviso(values, weights), expected, rtol=0, atol=5e-8)


def test_biviso_matches_unrounded_cran_fixtures():
    groups = {}
    path = Path(__file__).parent / "fixtures" / "keyboard-combination-biviso.csv"
    with path.open() as source:
        for row in csv.DictReader(source):
            groups.setdefault(row["case"], []).append(row)
    assert len(groups) == 16
    for case, rows in groups.items():
        shape = (int(rows[0]["nrow"]), int(rows[0]["ncol"]))
        values = np.array([float(r["raw_estimate"]) for r in rows]).reshape(shape)
        weights = np.array([float(r["weight"]) for r in rows]).reshape(shape)
        expected = np.array([float(r["biviso_fit"]) for r in rows]).reshape(shape)
        assert_allclose(_biviso(values, weights), expected, rtol=0, atol=5e-8, err_msg=case)


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


def test_keyboard_combination_key2_and_key3_diagonal_candidate_contracts():
    patients = np.zeros((2, 3), dtype=int)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 1
    excluded = np.zeros_like(patients, dtype=bool)
    excluded[0, 1] = True
    excluded[1, 0] = True

    key1 = KeyboardCombDesign(early_stop_patients=None)
    key2 = KeyboardCombDesign(early_stop_patients=None, movement_algorithm="key2")
    key3 = KeyboardCombDesign(early_stop_patients=None, movement_algorithm="key3")
    assert key1.next_dose(patients, toxicities, (1, 1), eliminated=excluded).next_dose == (1, 1)
    key2_up = key2.next_dose(patients, toxicities, (1, 1), eliminated=excluded)
    assert key2_up.action == "stay"
    assert key2_up.candidate_doses == ()
    key3_up = key3.next_dose(patients, toxicities, (1, 1), eliminated=excluded)
    assert key3_up.action == "escalate"
    assert key3_up.next_dose == (2, 2)
    assert key3_up.candidate_doses == ((2, 2),)
    np.testing.assert_array_equal(key3_up.candidate_probabilities, [1.0])

    # On de-escalation, key2 includes the diagonal predecessor while key1 does not.
    patients[:] = 0
    toxicities[:] = 0
    patients[1, 1] = 1
    toxicities[1, 1] = 1
    excluded[:] = False
    excluded[0, 1] = True
    excluded[1, 0] = True
    assert key1.next_dose(patients, toxicities, (2, 2), eliminated=excluded).next_dose == (2, 2)
    key2_down = key2.next_dose(patients, toxicities, (2, 2), eliminated=excluded)
    assert key2_down.action == "deescalate"
    assert key2_down.next_dose == (1, 1)


def test_keyboard_combination_key4_uses_raw_beta_one_candidate_masses():
    design = KeyboardCombDesign(early_stop_patients=None, movement_algorithm="key4")
    patients = np.zeros((2, 3), dtype=int)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 1
    patients[1, 0] = 10
    toxicities[1, 0] = 3
    candidates = ((2, 1), (1, 2))
    low, high = design.target - design.margin_left, design.target + design.margin_right
    masses = np.array(
        [
            betainc(4, 8, high) - betainc(4, 8, low),
            high - low,  # Empty cell has the paper's Beta(1,1) prior.
        ]
    )
    expected = masses / masses.sum()
    rng_seed = 903
    expected_choice = candidates[int(np.random.default_rng(rng_seed).choice(2, p=expected))]
    step = design.next_dose(patients, toxicities, (1, 1), rng=rng_seed)
    assert step.action == "escalate"
    assert step.candidate_doses == candidates
    np.testing.assert_allclose(step.candidate_probabilities, expected, rtol=0, atol=1e-15)
    assert step.next_dose == expected_choice
    assert step.candidate_probabilities is not None
    assert not step.candidate_probabilities.flags.writeable


def test_keyboard_combination_variants_match_independent_base_r_fixtures():
    fixture_dir = Path(__file__).parent / "fixtures"
    with (fixture_dir / "keyboard-combination-variants-config.csv").open() as stream:
        configs = {row["case"]: row for row in csv.DictReader(stream)}
    with (fixture_dir / "keyboard-combination-variants-decisions.csv").open() as stream:
        decisions = {(row["case"], row["algorithm"]): row for row in csv.DictReader(stream)}
    with (fixture_dir / "keyboard-combination-variants-masses.csv").open() as stream:
        mass_rows = list(csv.DictReader(stream))

    for (case, algorithm), expected in decisions.items():
        config = configs[case]
        patients = np.fromstring(config["patients_row_major"], sep=";").reshape(3, 3)
        toxicities = np.fromstring(config["toxicities_row_major"], sep=";").reshape(3, 3)
        excluded = np.fromstring(config["excluded_row_major"], sep=";").reshape(3, 3).astype(bool)
        current = (int(config["current_a"]), int(config["current_b"]))
        target = float(config["target"])
        design = KeyboardCombDesign(
            target=target,
            margin_left=target - float(config["lower"]),
            margin_right=float(config["upper"]) - target,
            early_stop_patients=None,
            movement_algorithm=algorithm,
        )
        result = design.next_dose(patients, toxicities, current, eliminated=excluded, rng=421)
        expected_doses = tuple(
            tuple(int(part) for part in item.split(","))
            for item in expected["eligible_candidates"].split(";")
        )
        selected = {
            tuple(int(part) for part in item.split(","))
            for item in expected["selected_candidates"].split(";")
        }
        assert result.action == expected["movement"]
        assert result.candidate_doses == expected_doses
        assert result.candidate_probabilities is not None
        expected_probabilities = np.array(
            [
                float(
                    next(
                        row["selection_probability"]
                        for row in mass_rows
                        if row["case"] == case
                        and row["algorithm"] == algorithm
                        and (int(row["candidate_a"]), int(row["candidate_b"])) == dose
                    )
                )
                for dose in expected_doses
            ]
        )
        np.testing.assert_allclose(
            result.candidate_probabilities,
            expected_probabilities,
            rtol=0,
            atol=1e-12,
        )
        assert result.next_dose in selected


def test_keyboard_combination_default_key1_matches_explicit_option_and_boundaries():
    implicit = KeyboardCombDesign(early_stop_patients=None)
    explicit = KeyboardCombDesign(early_stop_patients=None, movement_algorithm="key1")
    patients = np.zeros((2, 3), dtype=int)
    toxicities = np.zeros_like(patients)
    patients[0, 0] = 3
    implicit_step = implicit.next_dose(patients, toxicities, (1, 1), rng=77)
    explicit_step = explicit.next_dose(patients, toxicities, (1, 1), rng=77)
    assert implicit_step.action == explicit_step.action
    assert implicit_step.next_dose == explicit_step.next_dose
    toxicities[0, 0] = 1
    boundary_stay = implicit.next_dose(patients, toxicities, (1, 1), rng=77)
    assert boundary_stay.action == "stay"
    assert boundary_stay.next_dose == (1, 1)
    edge = KeyboardCombDesign(early_stop_patients=None, movement_algorithm="key3")
    patients[:] = 0
    toxicities[:] = 0
    patients[1, 2] = 1
    boundary_step = edge.next_dose(patients, toxicities, (2, 3), rng=77)
    assert boundary_step.action == "stay"
    assert boundary_step.next_dose == (2, 3)


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
    assert empty.dose is None
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


def test_low_target_boundaries_match_reference_guard_and_cutoffs():
    design = KeyboardCombDesign(
        target=0.05,
        margin_left=0.02,
        margin_right=0.02,
        early_stop_patients=None,
    )
    assert design._cutoffs(3) == (0, 1, 1)
    assert design._cutoffs(6) == (0, 1, 1)


def test_eliminated_current_dose_never_falls_back_to_reenrollment():
    design = KeyboardCombDesign(early_stop_patients=None)
    patients = np.array([[3, 3], [3, 3]])
    toxicities = np.array([[0, 3], [3, 3]])
    step = design.next_dose(patients, toxicities, (2, 2))
    assert step.action == "stop_safety"
    assert step.next_dose is None
