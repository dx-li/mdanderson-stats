"""R reference results and independent beta/binomial interval probabilities."""

import csv
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import KeyboardDesign, simulate_keyboard

FIXTURES = Path(__file__).parent / "fixtures"


def test_equal_width_and_rescaled_endpoint_keys():
    design = KeyboardDesign(0.2)
    assert_allclose(design.intervals[:, 0], np.arange(1, 18, 2) / 20)
    assert_allclose(design.intervals[:, 1], np.arange(3, 20, 2) / 20)
    assert design.target_key == 2
    result = design.posterior_keys(3, 0)
    assert result.probability.sum() < 1
    assert int(result.move) == 1
    extended = KeyboardDesign(0.2, edge_rule="rescale").posterior_keys(3, 0)
    assert_allclose(extended.probability.sum(), 1)
    assert_allclose(extended.score[0], 2 * extended.probability[0])
    extreme = design.posterior_keys([200, 200], [0, 200])
    assert_array_equal(extreme.move, [1, -1])


def test_interval_probabilities_against_rational_binomial_identity():
    design = KeyboardDesign(0.2)
    for n in range(1, 13):
        for y in range(n + 1):

            def cdf(x):
                return sum(
                    Fraction(comb(n + 1, k)) * x**k * (1 - x) ** (n + 1 - k)
                    for k in range(y + 1, n + 2)
                )

            exact = [
                float(cdf(Fraction(k + 2, 20)) - cdf(Fraction(k, 20))) for k in range(1, 18, 2)
            ]
            assert_allclose(design.posterior_keys(n, y).probability, exact, atol=2e-15, rtol=3e-13)


def test_native_boundary_tables_and_safety_minimum():
    for edge in ["discard", "rescale"]:
        tables = {
            p: KeyboardDesign(p, edge_rule=edge).boundary_table(30)
            for p in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        }
        with (FIXTURES / "keyboard-boundaries.csv").open() as handle:
            for row in csv.DictReader(handle):
                n = int(row["patients"])
                table = tables[float(row["target"])]
                assert table.escalate_max[n - 1] == int(row["escalate"])
                assert table.deescalate_min[n - 1] == int(row["deescalate"])
                if n >= 3:
                    expected = n + 1 if row["eliminate"] == "NA" else int(row["eliminate"])
                    assert table.eliminate_min[n - 1] == expected
    assert_array_equal(KeyboardDesign(0.3).boundary_table(2).eliminate_min, [2, 3])


def test_native_mtd_selections_and_reporting():
    with (FIXTURES / "keyboard-selection.csv").open() as handle:
        for row in csv.DictReader(handle):
            design = KeyboardDesign(float(row["target"]), extra_safe=row["extra_safe"] == "TRUE")
            n = np.array(row["patients"].split(";"), int)
            y = np.array(row["toxicities"].split(";"), int)
            result = design.select_mtd(n, y)
            expected = int(row["mtd"])
            assert result.dose == (None if expected == 99 else expected)
            estimates = [np.nan if x == "----" else float(x) for x in row["estimate"].split(";")]
            assert_allclose(
                result.isotonic_mean, estimates, atol=0.005000000001, rtol=0, equal_nan=True
            )


def test_native_simulation_and_nondefault_key_forwarding():
    reference = np.genfromtxt(FIXTURES / "keyboard-oc.csv", delimiter=",", names=True)
    result = simulate_keyboard(
        KeyboardDesign(), [0.05, 0.15, 0.3, 0.45, 0.6], trials=10000, rng=127
    )
    expected = reference["selection_probability"]
    error = np.sqrt(result.selection_mcse[1:] ** 2 + expected * (1 - expected) / 10000)
    assert np.all(np.abs(result.selection_probability[1:] - expected) < 6 * error)
    assert_allclose(result.mean_patients, reference["mean_patients"], atol=0.5, rtol=0)
    assert_allclose(result.mean_toxicities, reference["mean_toxicities"], atol=0.15, rtol=0)
    # With no lower key, zero toxicity stays in the target key rather than escalating.
    wide = simulate_keyboard(KeyboardDesign(lower=0, upper=0.5), [0, 0, 0], trials=2, rng=1)
    assert_array_equal(wide.patients, [[30, 0, 0]] * 2)
    assert_array_equal(wide.selected_dose, [1, 1])


def test_safety_precedence_and_unconditional_precision_stop():
    design = KeyboardDesign(early_stop_patients=12)
    assert design.next_dose([12, 0], [0, 0], 1).action == "stop_precision"
    assert design.next_dose([3, 12], [0, 12], 2).next_dose == 1
    assert KeyboardDesign(extra_safe=True).next_dose([3, 0], [2, 0], 1).action == "stop_safety"
    result = simulate_keyboard(KeyboardDesign(early_stop_patients=3), [1, 1], trials=2)
    assert result.stop_reason == ("stop_safety", "stop_safety")
    assert_array_equal(result.selected_dose, [0, 0])
