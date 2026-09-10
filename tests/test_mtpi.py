from fractions import Fraction
from math import comb

import numpy as np
import pytest

from mdanderson_stats import MTPIDesign, simulate_mtpi


def beta_cdf_integer(x, a, b):
    degree = a + b - 1
    return float(
        sum(
            Fraction(comb(degree, k)) * x**k * (1 - x) ** (degree - k) for k in range(a, degree + 1)
        )
    )


def test_posterior_matches_exact_polynomial_probabilities_and_tails():
    design = MTPIDesign()
    for n, y in [(1, 0), (3, 1), (6, 4), (30, 13), (200, 0), (200, 200)]:
        result = design.posterior(n, y)
        a, b = y + 1, n - y + 1
        lo = beta_cdf_integer(Fraction(1, 4), a, b)
        # Reflect the beta variable to compute the upper tail independently.
        hi = beta_cdf_integer(Fraction(13, 20), b, a)
        np.testing.assert_allclose(result.probability[[0, 2]], [lo, hi], rtol=2e-13, atol=0)
        assert result.probability.sum() == pytest.approx(1, abs=3e-15)
        np.testing.assert_allclose(
            result.unit_probability_mass, result.probability / [0.25, 0.1, 0.65], rtol=1e-14
        )
        assert result.overdose_probability == pytest.approx(
            beta_cdf_integer(Fraction(7, 10), b, a), rel=2e-13, abs=0
        )


def test_public_excel_screenshot_decisions():
    table = MTPIDesign().decision_table(30).action
    # Screenshot: target .30, epsilon1=epsilon2=.05, N=30.
    assert list(table[:4, 2]) == ["E", "S", "D", "DU"]
    assert list(table[:7, 5]) == ["E", "E", "S", "S", "DU", "DU", "DU"]
    assert list(table[:13, 11]) == ["E"] * 3 + ["S"] * 3 + ["D"] + ["DU"] * 6
    assert list(table[:, 29]) == ["E"] * 7 + ["S"] * 7 + ["DU"] * 17
    assert table[2, 0] == ""
    assert not table.flags.writeable


def test_paper_safety_rules_no_three_patient_gate_and_sticky_exclusion():
    design = MTPIDesign()
    assert design.next_dose([2, 0], [2, 0], 1).action == "stop_safety"
    # A toxic higher dose causes D, then is barred when a later E attempts return.
    first = design.next_dose([3, 3, 0], [0, 3, 0], 2)
    assert first.next_dose == 1
    assert not first.eliminated.any()
    second = design.next_dose([6, 3, 0], [0, 3, 0], 1, eliminated=first.eliminated)
    assert second.next_dose == 1
    np.testing.assert_array_equal(second.eliminated, [False, True, True])
    third = design.next_dose([9, 3, 0], [0, 3, 0], 1, eliminated=second.eliminated)
    assert third.next_dose == 1


def test_isotonic_selection_and_paper_tie_preference():
    design = MTPIDesign(target=0.5, lower=0.45, upper=0.55)
    # Posterior means .25 and .75: equal distance prefers the lower safe-side dose.
    assert design.select_mtd([2, 2], [0, 2]).dose == 1
    # Pool violations explicitly: means .5, .25 => equal-weight pooled .375.
    result = design.select_mtd([2, 2, 0], [1, 0, 0])
    np.testing.assert_allclose(result.isotonic_mean[:2], [0.375, 0.375])
    assert result.dose == 2
    weighted = design.select_mtd([2, 2], [1, 0], weights=[3, 1])
    np.testing.assert_allclose(weighted.isotonic_mean, [0.4375, 0.4375])
    huge = design.select_mtd([2, 2], [1, 0], weights=[1e308, 1e308])
    np.testing.assert_allclose(huge.isotonic_mean, [0.375, 0.375])
    assert MTPIDesign().select_mtd([2, 0], [2, 0]).dose is None


def test_simulation_against_exhaustive_response_paths():
    design = MTPIDesign()
    p = np.array([0.2, 0.45])
    probabilities = np.zeros(3)
    expected_n = np.zeros(2)

    def visit(n, y, dose, excluded, weight, remaining):
        if remaining == 0 or dose is None:
            selected = design.select_mtd(n, y, eliminated=excluded).dose
            probabilities[0 if selected is None else selected] += weight
            expected_n[:] += weight * n
            return
        j = dose - 1
        for event in [0, 1]:
            next_n, next_y = n.copy(), y.copy()
            next_n[j] += 1
            next_y[j] += event
            step = design.next_dose(next_n, next_y, dose, eliminated=excluded)
            visit(
                next_n,
                next_y,
                step.next_dose,
                step.eliminated,
                weight * (p[j] if event else 1 - p[j]),
                remaining - 1,
            )

    visit(np.zeros(2, dtype=int), np.zeros(2, dtype=int), 1, np.zeros(2, dtype=bool), 1.0, 5)
    result = simulate_mtpi(design, p, cohorts=5, cohort_size=1, trials=12000, rng=72)
    assert probabilities.sum() == pytest.approx(1)
    np.testing.assert_array_less(
        abs(result.selection_probability - probabilities),
        5 * np.sqrt(probabilities * (1 - probabilities) / 12000) + 1e-12,
    )
    np.testing.assert_allclose(result.mean_patients, expected_n, atol=0.04)
    assert np.all(result.toxicities <= result.patients)
    repeat = simulate_mtpi(design, p, cohorts=5, cohort_size=1, trials=12000, rng=72)
    np.testing.assert_array_equal(result.patients, repeat.patients)
    np.testing.assert_array_equal(result.selected_dose, repeat.selected_dose)
    toxic = simulate_mtpi(design, [1, 1], trials=3, rng=1)
    assert toxic.stopped_safety.all()
    np.testing.assert_array_equal(toxic.patients, [[3, 0]] * 3)
    safe = simulate_mtpi(design, [0, 0], trials=3, rng=1)
    np.testing.assert_array_equal(safe.selected_dose, [2, 2, 2])
