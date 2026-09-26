"""Compare independent Python formulas with original STPLAN 4.5 routines."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_continuous import (
    stplan_exponential_one_sample_power,
    stplan_exponential_two_sample_power,
    stplan_lognormal_two_sample_power,
    stplan_normal_one_sample_power,
    stplan_normal_two_sample_power,
    stplan_welch_two_sample_power,
)
from mdanderson_stats.stplan_correlation import (
    stplan_correlation_one_sample_power,
    stplan_correlation_two_sample_power,
)


def test_continuous_power_against_original_fortran():
    with (Path(__file__).parent / "fixtures" / "stplan-continuous.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    actual, expected = [], []
    for row in rows:
        a = [float(row[f"a{i}"]) for i in range(1, 7)]
        match row["routine"]:
            case "pnor1":
                value = stplan_normal_one_sample_power(*a[:3], alpha=a[3])
            case "pnev2":
                value = stplan_normal_two_sample_power(a[0], a[4], a[2], a[3], alpha=a[1])
            case "pnuv2":
                value = stplan_welch_two_sample_power(a[0], a[4], a[5], a[2], a[3], alpha=a[1])
            case "lognormal":
                value = stplan_lognormal_two_sample_power(*a[:5], alpha=a[5])
            case "pexp1":
                value = stplan_exponential_one_sample_power(*a[:2], alpha=a[2])
            case "pexp2":
                # Native arguments are larger/smaller mean ratio and F degrees
                # of freedom, twice the corresponding sample sizes.
                value = stplan_exponential_two_sample_power(a[0], 1, a[1] / 2, a[2] / 2, alpha=a[3])
            case "pcor1":
                value = stplan_correlation_one_sample_power(*a[:3], alpha=a[3])
            case "pcor2":
                value = stplan_correlation_two_sample_power(*a[:4], alpha=a[4])
            case _:
                raise AssertionError(f"Unknown reference routine: {row['routine']}")
        actual.append(float(value))
        expected.append(float(row["power"]))
    # STPLAN's original quantile solvers request 1e-7 absolute/relative
    # tolerance. The modern SciPy evaluation need not copy their rounding.
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
