"""Native numerical kernels and exact small-trial references for PoPdesign."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.pop_design import PoPDesign, predictive_bayes_factor
from mdanderson_stats.pop_simulation import simulate_pop


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_full_boundaries_and_predictive_factors_against_native_r():
    rows = _rows("pop-boundaries.csv")
    for phi in (0.05, 0.15, 0.25, 0.3, 0.4, 0.6):
        reference = [r for r in rows if float(r["target"]) == phi]
        table = PoPDesign(phi).boundaries(60)
        for actual, column, lower in (
            (table.escalate_max, "escalate", True),
            (table.deescalate_min, "deescalate", False),
            (table.exclude_under_max, "exclude_under", True),
            (table.exclude_over_min, "exclude_over", False),
        ):
            expected = [
                int(r[column]) if r[column] != "NA" else (-1 if lower else int(r["patients"]) + 1)
                for r in reference
            ]
            np.testing.assert_array_equal(actual, expected)
    for row in _rows("pop-factor.csv"):
        actual = predictive_bayes_factor(
            float(row["target"]), int(row["patients"]), int(row["toxicities"]), log=True
        )
        np.testing.assert_allclose(actual, float(row["log_bayes_factor"]), atol=2e-12, rtol=0)
    assert predictive_bayes_factor(0.05, 1000, 1000, log=True) < -2000


def test_weighted_isotonic_selection_against_native_r():
    for row in _rows("pop-selection.csv"):
        result = PoPDesign(float(row["target"])).select_mtd(
            [int(x) for x in row["patients"].split(";")],
            [int(x) for x in row["toxicities"].split(";")],
        )
        assert (result.dose or 0) == int(row["selected"])
        if row["estimates"]:
            np.testing.assert_allclose(
                result.isotonic_estimate[result.eligible],
                [float(x) for x in row["estimates"].split(";")],
                atol=2e-14,
                rtol=0,
            )


def test_paper_safety_minimum_and_untried_dose_labels():
    assert PoPDesign(0.2).select_mtd([1, 0], [1, 0]).dose == 1
    assert PoPDesign(0.2, safety_min_patients=0).select_mtd([1, 0], [1, 0]).dose is None
    untried = PoPDesign(0.3).select_mtd([3, 0, 3], [0, 0, 1])
    assert untried.dose == 3
    assert np.isnan(untried.isotonic_estimate[1])
    np.testing.assert_allclose(
        untried.isotonic_estimate[[0, 2]],
        np.array([0.05 / 3.1, 1.05 / 3.1]) + np.array([1, 2]) * 1e-10,
        atol=1e-15,
        rtol=0,
    )


def test_simulation_against_exact_cohort_outcome_enumeration():
    for row in _rows("pop-exact-oc.csv"):
        p = [float(row["p1"]), float(row["p2"])]
        trials = 2000 if p[0] not in (0, 1) else 2
        result = simulate_pop(
            PoPDesign(0.3),
            p,
            total_patients=9,
            cohort_size=3,
            titration=False,
            risk_cutoff=0.5,
            trials=trials,
            seed=175,
        )
        expected = np.array([float(row[k]) for k in ("no_mtd", "dose1", "dose2")])
        assert np.all(
            abs(result.selection_probability - expected)
            <= 6 * np.sqrt(expected * (1 - expected) / trials) + 0.002
        )
        for data, columns in (
            (result.patients, ("mean_n1", "mean_n2")),
            (result.toxicities, ("mean_y1", "mean_y2")),
        ):
            error = abs(data.mean(axis=0) - [float(row[c]) for c in columns])
            assert np.all(error <= 6 * data.std(axis=0) / np.sqrt(trials) + 0.005)
        for actual, key in (
            (result.early_stop.mean(), "early"),
            (result.risk_under, "risk_under"),
            (result.risk_over, "risk_over"),
        ):
            q = float(row[key])
            assert abs(actual - q) <= 6 * np.sqrt(q * (1 - q) / trials) + 0.002
