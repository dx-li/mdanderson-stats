"""Independent R quadrature/sums and original STPLAN survival references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_survival import (
    stplan_censored_exponential_one_sample_power,
    stplan_george_desu_survival_power,
    stplan_historical_survival_power,
    stplan_information_survival_power,
    stplan_piecewise_survival_power,
)


def _evaluate(row):
    a = [float(row[f"a{i}"]) for i in range(1, 11)]
    match row["routine"]:
        case "one":
            return stplan_censored_exponential_one_sample_power(*a[:5], alpha=a[5])
        case "george_desu":
            return stplan_george_desu_survival_power(*a[:5], alpha=a[5])
        case "information":
            return stplan_information_survival_power(*a[:5], alpha=a[5])
        case "historical":
            return stplan_historical_survival_power(
                a[0],
                a[5],
                a[1],
                a[2],
                a[3],
                a[6],
                a[7],
                control_allocation=a[8],
                continued_followup=bool(a[9]),
                alpha=a[4],
            )
        case "piecewise_lower" | "piecewise_higher":
            return stplan_piecewise_survival_power(
                a[6],
                a[7],
                a[5],
                a[0],
                a[1],
                a[2],
                a[3],
                model_arm="lower_hazard"
                if row["routine"] == "piecewise_lower"
                else "higher_hazard",
                alpha=a[4],
            )
        case _:
            raise AssertionError(f"Unknown reference routine: {row['routine']}")


def test_survival_against_independent_r_quadrature_and_sums():
    path = Path(__file__).parent / "fixtures" / "stplan-survival-r.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        # Only the Poisson-event mixture truncates: its default omitted mass
        # is <=1e-10. R's wider finite sum leaves <1e-14 probability.
        tolerance = 1.1e-10 if row["routine"] == "one" else 2e-12
        np.testing.assert_allclose(_evaluate(row), float(row["power"]), rtol=0, atol=tolerance)
        assert float(row["omitted"]) < 1e-14


def test_survival_against_original_fortran_where_formula_is_valid():
    path = Path(__file__).parent / "fixtures" / "stplan-survival-native.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["routine"] == "piecewise_native":
            # Native piecewise normalization is defective, including two NaNs.
            # Independent quadrature above is the correctness reference.
            continue
        tolerance = {"one": 1.1e-7, "george_desu": 2e-8}.get(row["routine"], 2e-13)
        np.testing.assert_allclose(_evaluate(row), float(row["power"]), rtol=0, atol=tolerance)
