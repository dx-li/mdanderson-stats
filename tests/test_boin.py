"""Published boundaries, rational posterior safety checks and native R comparisons."""

import csv
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import BOINDesign, simulate_boin


def test_published_boundaries_and_cohort_table():
    for target, lower, upper in zip(
        [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        [0.118, 0.157, 0.197, 0.236, 0.276, 0.316],
        [0.179, 0.238, 0.298, 0.359, 0.419, 0.480],
        strict=True,
    ):
        design = BOINDesign(target)
        assert_allclose(
            [design.escalation_boundary, design.deescalation_boundary],
            [lower, upper],
            atol=0.0005,
            rtol=0,
        )
    table = BOINDesign(0.3).boundary_table()
    assert_array_equal(table.escalate_max[2::3], [0, 1, 2, 2, 3, 4, 4, 5, 6, 7])
    assert_array_equal(table.deescalate_min[2::3], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert_array_equal(table.eliminate_min[2::3], [3, 4, 5, 7, 8, 9, 10, 11, 12, 14])


def test_safety_cutoffs_against_rational_binomial_identity():
    table = BOINDesign(0.3, extra_safe=True).boundary_table()
    phi = Fraction(3, 10)
    for n in range(1, 31):
        for cutoff, boundaries in [
            (Fraction(19, 20), table.eliminate_min),
            (Fraction(9, 10), table.lowest_stop_min),
        ]:
            expected = n + 1
            if n >= 3:
                for y in range(n + 1):
                    tail = sum(
                        Fraction(comb(n + 1, k)) * phi**k * (1 - phi) ** (n + 1 - k)
                        for k in range(y + 1)
                    )
                    if tail > cutoff:
                        expected = y
                        break
            assert boundaries[n - 1] == expected


def test_native_r_selection_and_report_estimates():
    with (Path(__file__).parent / "fixtures/boin-reference.csv").open() as handle:
        for row in csv.DictReader(handle):
            n = np.array(row["patients"].split(";"), int)
            y = np.array(row["toxicities"].split(";"), int)
            design = BOINDesign(
                float(row["target"]),
                extra_safe=row["extra_safe"] == "TRUE",
                bound_mtd=row["bound_mtd"] == "TRUE",
            )
            result = design.select_mtd(n, y)
            expected = int(row["mtd"])
            assert result.dose == (None if expected == 99 else expected)
            estimates = np.array(
                [np.nan if v == "----" else float(v) for v in row["estimate"].split(";")]
            )
            assert_allclose(
                result.isotonic_mean, estimates, atol=0.005000000001, rtol=0, equal_nan=True
            )


def test_safety_history_and_extra_lowest_rule():
    design = BOINDesign(0.3)
    assert design.next_dose([3, 0], [2, 0], 1).action == "stay"
    assert BOINDesign(0.3, extra_safe=True).next_dose([3, 0], [2, 0], 1).action == "stop_safety"
    stopped = design.next_dose([3, 0], [3, 0], 1)
    assert stopped.next_dose is None and np.all(stopped.eliminated)
    assert design.select_mtd([3, 0], [3, 0]).dose is None
    # Exclusions cannot be undone by a later snapshot with a lower observed rate.
    decision = design.next_dose([6, 3, 0], [0, 0, 0], 1, eliminated=[False, True, True])
    assert decision.next_dose == 1
    assert design.select_mtd([6, 3, 0], [0, 0, 0], eliminated=decision.eliminated).dose == 1


def test_rule_modifications_and_stay_only_precision_stop():
    assert BOINDesign(0.25).next_dose([3, 3, 0], [0, 1, 0], 2).next_dose == 1
    assert (
        BOINDesign(0.25, stay_at_one_of_three=True).next_dose([3, 3, 0], [0, 1, 0], 2).next_dose
        == 2
    )
    assert BOINDesign(0.3).next_dose([3, 6, 0], [0, 2, 0], 2).next_dose == 2
    assert (
        BOINDesign(0.3, deescalate_at_two_of_six=True).next_dose([3, 6, 0], [0, 2, 0], 2).next_dose
        == 1
    )
    design = BOINDesign(0.25, early_stop_patients=12)
    assert design.next_dose([3, 12, 0], [0, 3, 0], 2).action == "stop_precision"
    assert design.next_dose([3, 12, 0], [0, 0, 0], 2).next_dose == 3
    assert design.next_dose([3, 12], [0, 0], 2).action == "stay"


def test_weighted_pooling_and_tie_selection():
    n = np.array([3, 6, 6, 0])
    y = np.array([0, 3, 1, 0])
    mean = (y + 0.05) / (n + 0.1)
    variance = (y + 0.05) * (n - y + 0.05) / ((n + 0.1) ** 2 * (n + 1.1))
    pooled = np.sum(mean[1:3] / variance[1:3]) / np.sum(1 / variance[1:3])
    below = BOINDesign(0.5).select_mtd(n, y)
    assert_allclose(below.isotonic_mean[:3], [mean[0], pooled, pooled])
    assert below.dose == 3
    assert np.isnan(below.isotonic_mean[3])
    assert np.all(below.isotonic_interval[:3, 0] <= below.isotonic_mean[:3])
    assert BOINDesign(0.3).select_mtd([0, 0], [0, 0]).dose is None


def test_invalid_counts_and_unsupported_modifications():
    with pytest.raises(ValueError, match="y<=n"):
        BOINDesign().select_mtd([3, 0], [4, 0])
    with pytest.raises(ValueError, match="evaluated"):
        BOINDesign().next_dose([0, 0], [0, 0], 1)
    with pytest.raises(ValueError, match="1/3 modification"):
        BOINDesign(0.3, stay_at_one_of_three=True)


def test_simulation_deterministic_safety_and_escalation_paths():
    safe = simulate_boin(BOINDesign(0.3), [0, 0, 0], trials=2, rng=5)
    assert_array_equal(safe.patients, [[3, 3, 24]] * 2)
    assert_array_equal(safe.selected_dose, [3, 3])
    unsafe = simulate_boin(BOINDesign(0.3), [1, 1, 1], trials=2, rng=5)
    assert_array_equal(unsafe.patients, [[3, 0, 0]] * 2)
    assert_array_equal(unsafe.selected_dose, [0, 0])
    assert unsafe.safety_stop_probability == 1


def test_one_cohort_simulation_against_binomial_law():
    result = simulate_boin(BOINDesign(0.3), [0.4, 0.6], cohorts=1, trials=4000, rng=6)
    counts = np.bincount(result.toxicities[:, 0], minlength=4) / 4000
    exact = np.array([comb(3, y) * 0.4**y * 0.6 ** (3 - y) for y in range(4)])
    assert np.all(np.abs(counts - exact) < 6 * np.sqrt(exact * (1 - exact) / 4000))
    assert result.safety_stop_probability == counts[3]
    assert result.selection_probability[0] == counts[3]


def test_simulation_against_native_r_operating_characteristics():
    reference = np.genfromtxt(
        Path(__file__).parent / "fixtures/boin-oc-reference.csv", delimiter=",", names=True
    )
    result = simulate_boin(BOINDesign(0.3), [0.05, 0.15, 0.3, 0.45, 0.6], trials=10000, rng=120)
    expected = reference["selection_probability"]
    # Independent R and NumPy streams: combine the two Monte Carlo variances.
    error = np.sqrt(result.selection_mcse[1:] ** 2 + expected * (1 - expected) / 10000)
    assert np.all(np.abs(result.selection_probability[1:] - expected) < 6 * error)
    assert_allclose(result.mean_patients, reference["mean_patients"], atol=0.5, rtol=0)
    assert_allclose(result.mean_toxicities, reference["mean_toxicities"], atol=0.15, rtol=0)
