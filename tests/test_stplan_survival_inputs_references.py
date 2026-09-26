"""Survival-curve conversions checked against independent R linear systems."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_survival_inputs import stplan_piecewise_from_survival


def test_survival_curve_conversion_matches_independent_r():
    fixture = Path(__file__).parent / "fixtures" / "stplan-survival-inputs-r.csv"
    with fixture.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        count = int(row["points"])
        times = [float(row[f"t{i}"]) for i in range(1, count + 1)]
        survival = [float(row[f"s{i}"]) for i in range(1, count + 1)]
        change = float(row["change_time"]) if count == 2 else None
        result = stplan_piecewise_from_survival(times, survival, change_time=change)
        np.testing.assert_allclose(
            [result.hazard_before, result.hazard_after, result.change_time],
            [float(row[name]) for name in ("hazard_before", "hazard_after", "change_time")],
            rtol=1e-9,
            atol=1e-14,
            err_msg=row["case_id"],
        )
        assert bool(result.change_time_identified) == (row["identified"] == "TRUE")
        times = np.asarray(times)
        cumulative = result.hazard_before * np.minimum(times, result.change_time)
        cumulative += result.hazard_after * np.maximum(times - result.change_time, 0)
        np.testing.assert_allclose(cumulative, -np.log(survival), rtol=1e-12, atol=1e-13)
