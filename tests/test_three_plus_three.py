"""Independent selection probabilities and enrollment matching invariants."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import BOINDesign, compare_boin_three_plus_three, simulate_three_plus_three


def test_deterministic_escalation_confirmation_and_deescalation():
    safe = simulate_three_plus_three([0, 0, 0], trials=2, rng=6)
    assert_array_equal(safe.patients, [[3, 3, 6]] * 2)
    assert_array_equal(safe.selected_dose, [3, 3])
    unsafe = simulate_three_plus_three([1, 1, 1], trials=2, rng=6)
    assert_array_equal(unsafe.patients, [[3, 0, 0]] * 2)
    assert_array_equal(unsafe.selected_dose, [0, 0])
    down = simulate_three_plus_three([0, 1, 1], trials=2, start_dose=3, rng=6)
    assert_array_equal(down.patients, [[6, 3, 3]] * 2)
    assert_array_equal(down.selected_dose, [1, 1])


@pytest.mark.parametrize("p", [[0.1, 0.3], [0.3, 0.5], [0.6, 0.2]])
def test_two_dose_selection_against_analytical_path_probabilities(p):
    # a=P(0/3), b=P(1/3). A confirmed dose is safe with probability a^2+2ab.
    a = (1 - np.array(p)) ** 3
    b = 3 * np.array(p) * (1 - np.array(p)) ** 2
    s = a * a + 2 * a * b
    select1 = (1 - s[1]) * s[0]
    select2 = a[0] * (1 + b[0]) * s[1]
    exact = np.array([1 - select1 - select2, select1, select2])
    result = simulate_three_plus_three(p, trials=100000, rng=120)
    assert np.all(
        np.abs(result.selection_probability - exact) < 6 * np.sqrt(exact * (1 - exact) / 100000)
    )
    rows = np.flatnonzero(result.selected_dose)
    doses = result.selected_dose[rows] - 1
    assert np.all(result.patients[rows, doses] == 6)
    assert np.all(result.toxicities[rows, doses] <= 1)
    assert np.all(result.patients <= 6)


def test_comparison_matching_enrollment_and_expansion_preserves_selection():
    design = BOINDesign(0.3)
    fixed = compare_boin_three_plus_three(
        design, [0, 0, 0], trials=2, matching="expand_three_plus_three", rng=6
    )
    assert_array_equal(fixed.three_plus_three.dose_finding_patients, [[3, 3, 6]] * 2)
    assert_array_equal(fixed.three_plus_three.patients, [[3, 3, 24]] * 2)
    assert_array_equal(fixed.expansion_patients, [18, 18])
    matched = compare_boin_three_plus_three(
        design, [0, 0, 0], trials=2, cohort_size=4, matching="match_boin", rng=6
    )
    assert_array_equal(matched.boin_max_patients, [12, 12])
    assert_array_equal(matched.boin.patients.sum(axis=1), [12, 12])
    random = compare_boin_three_plus_three(
        design, [0.1, 0.3, 0.5], trials=500, matching="expand_three_plus_three", rng=7
    )
    original = simulate_three_plus_three([0.1, 0.3, 0.5], trials=500, rng=7)
    assert_array_equal(random.three_plus_three.selected_dose, original.selected_dose)
    assert_array_equal(random.three_plus_three.dose_finding_toxicities, original.toxicities)
    expected = np.where(
        (random.boin.selected_dose > 0) & (original.selected_dose > 0),
        np.maximum(0, random.boin.patients.sum(axis=1) - original.patients.sum(axis=1)),
        0,
    )
    assert_array_equal(random.expansion_patients, expected)
    capped = compare_boin_three_plus_three(
        design, [0.1, 0.3, 0.5], trials=500, cohort_size=4, matching="match_boin", rng=7
    )
    assert_array_equal(capped.boin_max_patients, ((original.patients.sum(axis=1) + 3) // 4) * 4)
    assert np.all(capped.boin.patients.sum(axis=1) <= capped.boin_max_patients)
