"""Independent BOIN boundaries used by the BF-BOIN reference implementation."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.boin import BOINDesign


def test_bf_boin_reference_boundaries():
    fixture = Path(__file__).parent / "fixtures" / "bf-boin-boundaries.csv"
    with fixture.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = BOINDesign(target=0.25).boundary_table(len(rows))
    for column, actual in (
        ("n", table.patients),
        ("escalate_if_dlt_le", table.escalate_max),
        ("deescalate_if_dlt_ge", table.deescalate_min),
        ("eliminate_if_dlt_ge", table.eliminate_min),
    ):
        # R uses NA when elimination is impossible; Python uses n + 1.
        expected = [int(row[column]) if row[column] != "NA" else int(row["n"]) + 1 for row in rows]
        np.testing.assert_array_equal(actual, expected)
