import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln

from mdanderson_stats import TPIDesign, simulate_tpi


def test_original_paper_table_one_resolves_multiplier_order():
    table = TPIDesign().decision_table(12).action
    for n, expected in [
        (3, ["E", "S", "D", "DU"]),
        (6, ["E", "S", "S", "D", "DU", "DU", "DU"]),
        (9, ["E", "E", "S", "S", "S", "D", "DU", "DU", "DU", "DU"]),
        (12, ["E", "E", "E", "S", "S", "S", "D"] + ["DU"] * 6),
    ]:
        assert list(table[: n + 1, n - 1]) == expected
    # The conflicting prose K-label convention changes actual clinical actions.
    assert TPIDesign(lower_sd=1, upper_sd=1.5).posterior(6, 1).move == 1
    assert TPIDesign().posterior(6, 1).move == 0


def test_small_shape_beta_tails_against_weighted_quadrature():
    design = TPIDesign()
    for n, y in [(3, 0), (6, 1), (200, 0), (200, 60), (200, 200)]:
        result = design.posterior(n, y)
        a, b = y + 0.005, n - y + 0.005

        def cdf(x, shape1, shape2):
            if x == 0:
                return 0.0
            return quad(
                lambda t: np.exp((shape2 - 1) * np.log1p(-t) - betaln(shape1, shape2)),
                0,
                x,
                weight="alg",
                wvar=(shape1 - 1, 0),
                epsabs=1e-13,
                epsrel=1e-11,
            )[0]

        np.testing.assert_allclose(
            result.probability[[0, 2]],
            [cdf(result.lower, a, b), cdf(1 - result.upper, b, a)],
            rtol=2e-10,
            atol=1e-13,
        )
        np.testing.assert_allclose(result.probability.sum(), 1, atol=2e-15)
        assert result.standard_deviation == pytest.approx(
            np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        )
    clipped = TPIDesign(lower_sd=100, upper_sd=100).posterior(3, 1)
    np.testing.assert_array_equal(clipped.probability, [0, 1, 0])
    assert clipped.move == 0


def test_two_patient_safety_gate_permanent_exclusion_and_selection():
    design = TPIDesign()
    assert not design.posterior(1, 1).unsafe
    assert design.next_dose([1, 0], [1, 0], 1).next_dose == 1
    assert design.next_dose([2, 0], [2, 0], 1).action == "stop_safety"
    step = design.next_dose([3, 3, 0], [0, 3, 0], 2)
    assert step.next_dose == 1
    np.testing.assert_array_equal(step.eliminated, [False, True, False])
    assert design.next_dose([6, 3, 0], [0, 3, 0], 1, eliminated=step.eliminated).next_dose == 1
    # Explicit custom prior and weights affect inference, not just display.
    custom = TPIDesign(target=0.5, prior=(1, 1))
    selection = custom.select_mtd([2, 2, 0], [1, 0, 0], weights=[3, 1, 1])
    np.testing.assert_allclose(selection.isotonic_mean[:2], [0.4375, 0.4375])
    assert selection.dose == 2
    assert np.isnan(selection.isotonic_mean[2])


def test_batched_trials_match_exhaustive_paths():
    design = TPIDesign()
    p = np.array([0.2, 0.45])
    expected = np.zeros(3)
    expected_n = np.zeros(2)

    def visit(n, y, dose, excluded, probability, remaining):
        if remaining == 0 or dose is None:
            selected = None if dose is None else design.select_mtd(n, y, eliminated=excluded).dose
            expected[0 if selected is None else selected] += probability
            expected_n[:] += probability * n
            return
        j = dose - 1
        for event in [0, 1]:
            nn, yy = n.copy(), y.copy()
            nn[j] += 1
            yy[j] += event
            step = design.next_dose(nn, yy, dose, eliminated=excluded)
            visit(
                nn,
                yy,
                step.next_dose,
                step.eliminated,
                probability * (p[j] if event else 1 - p[j]),
                remaining - 1,
            )

    visit(np.zeros(2, dtype=int), np.zeros(2, dtype=int), 1, np.zeros(2, dtype=bool), 1.0, 6)
    result = simulate_tpi(design, p, cohorts=6, cohort_size=1, trials=20000, rng=72)
    np.testing.assert_allclose(result.selection_probability, expected, atol=0.012)
    np.testing.assert_allclose(result.mean_patients, expected_n, atol=0.04)
    toxic = simulate_tpi(design, [1, 1], cohorts=6, cohort_size=1, trials=3, rng=72)
    np.testing.assert_array_equal(toxic.patients, [[2, 0]] * 3)
    assert toxic.stopped_safety.all()
    safe = simulate_tpi(design, [0, 0], trials=3, rng=72)
    np.testing.assert_array_equal(safe.selected_dose, [2, 2, 2])
