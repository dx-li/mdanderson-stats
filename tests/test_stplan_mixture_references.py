"""Independent finite-sum references and native case-control comparisons."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_case_control import (
    stplan_case_control_power,
    stplan_matched_case_control_power,
)
from mdanderson_stats.stplan_poisson import stplan_poisson_two_sample_power


def test_case_control_and_poisson_against_independent_r_sums():
    path = Path(__file__).parent / "fixtures" / "stplan-mixtures-r.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    actual, expected = [], []
    for row in rows:
        a = [float(row[f"a{i}"]) for i in range(1, 7)]
        match row["routine"]:
            case "unmatched":
                value = stplan_case_control_power(*a[:5], alpha=a[5])
            case "matched":
                value = stplan_matched_case_control_power(*a[:4], alpha=a[4])
            case "poisson":
                value = stplan_poisson_two_sample_power(*a[:4], alpha=a[4])
            case _:
                raise AssertionError(f"Unknown reference routine: {row['routine']}")
        actual.append(float(value))
        expected.append(float(row["power"]))
        assert float(row["omitted"]) < 1e-14
    # Python truncates the infinite Poisson mixture at default omitted mass
    # <=1e-10. The independent R reference retains substantially more mass.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.1e-10)


def test_unmatched_power_against_original_fortran():
    path = Path(__file__).parent / "fixtures" / "stplan-case-control-native.csv"
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["routine"] == "unmatched"]
    actual, expected = [], []
    for row in rows:
        a = [float(row[f"a{i}"]) for i in range(1, 7)]
        actual.append(float(stplan_case_control_power(*a[:5], alpha=a[5])))
        expected.append(float(row["power"]))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)
