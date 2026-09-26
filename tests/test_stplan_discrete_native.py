"""Native-reference checks for STPLAN binary and count planning methods."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_discrete import (
    stplan_arcsine_binomial_two_sample_power,
    stplan_binomial_k_sample_power,
    stplan_exact_binomial_power,
    stplan_exact_poisson_power,
    stplan_fisher_exact_approx_power,
    stplan_historical_binomial_power,
    stplan_median_split_power,
    stplan_responder_normal_approximation_power,
    stplan_retention_probability,
)


def test_discrete_methods_against_original_fortran():
    with (Path(__file__).parent / "fixtures" / "stplan-discrete.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    actual, expected = [], []
    for row in rows:
        a = [float(row[f"a{i}"]) for i in range(1, 9)]
        match row["routine"]:
            case "arcsine":
                value = stplan_arcsine_binomial_two_sample_power(
                    *a[:4], alpha=a[4], sides=int(a[5])
                )
            case "median":
                value = stplan_median_split_power(*a[:3], alpha=a[3], sides=int(a[4]))
            case "historical":
                value = stplan_historical_binomial_power(*a[:4], alpha=a[4], sides=int(a[5]))
            case "responders":
                value = stplan_responder_normal_approximation_power(*a[:5], confidence=a[5])
            case "fisher":
                value = stplan_fisher_exact_approx_power(*a[:3], alpha=a[3])
            case "ksample2" | "ksample3":
                k = int(row["routine"][-1])
                value = stplan_binomial_k_sample_power(
                    a[:k], a[3 : 3 + k], alpha=a[6], sides=int(a[7])
                )
            case "retention":
                value = stplan_retention_probability(a[1], a[2], a[0], a[3])
            case "binomial":
                value = stplan_exact_binomial_power(*a[:3], alpha=a[3])
            case "poisson":
                value = stplan_exact_poisson_power(*a[:3], alpha=a[3])
            case _:
                raise AssertionError(f"Unknown reference routine: {row['routine']}")
        actual.append(float(value))
        native = float(row["power"])
        # The original exact-test routines return -1 when their rejection
        # region is empty. Python represents the corresponding valid power as 0.
        if native == -1:
            assert row["routine"] in {"binomial", "poisson"}
            native = 0
        expected.append(native)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
