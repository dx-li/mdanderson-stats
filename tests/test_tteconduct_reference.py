"""Independent historical-gamma integration and published conduct-table example."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats import (
    tteconduct_boundary_table,
    tteconduct_design,
    tteconduct_monitor,
)


def test_guide_boundary_and_probabilities_against_base_r():
    fixture = Path(__file__).parent / "fixtures" / "tteconduct-reference.csv"
    with fixture.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    design = tteconduct_design(60, 295, 3, 10, 1, 0.03, 40, max_total_time=4800)
    table = tteconduct_boundary_table(design, events=range(1, 7))
    roots = [row.minimum_total_time for row in table.boundaries]
    np.testing.assert_allclose(
        roots, [float(row["minimum_total_time"]) for row in reference], atol=1e-6, rtol=1e-7
    )
    np.testing.assert_array_equal(
        np.ceil(np.asarray(roots) * (365.25 / 12)), [0, 0, 105, 216, 330, 449]
    )
    for expected, boundary in zip(reference, table.boundaries, strict=True):
        assert not boundary.beyond_cap
        for time, column in ((1, "probability_at_one_month"), (20, "probability_at_twenty_months")):
            state = tteconduct_monitor(design, 10, int(expected["events"]), time)
            np.testing.assert_allclose(
                state.probability, float(expected[column]), atol=1e-9, rtol=0
            )
            assert state.stop_for_futility == (float(expected[column]) < 0.03)


def test_boundaries_preserve_time_units():
    roots = []
    for unit in (1e-100, 1.0, 1e100):
        design = tteconduct_design(4, 6 * unit, 2, unit, 0, 0.1, 10, max_total_time=100 * unit)
        table = tteconduct_boundary_table(design, events=[1, 4, 8])
        roots.append([row.minimum_total_time / unit for row in table.boundaries])
    np.testing.assert_allclose(roots, [roots[1]] * 3, rtol=1e-8, atol=1e-10)
