"""Native final-report mathematics; standalone report adapter is a separate contract."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import BOINDesign

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/single-reports.csv")
        .read_text()
        .splitlines()
    )
)


@pytest.mark.parametrize("row", ROWS)
def test_original_single_tite_report_numerics(row):
    n = np.fromstring(row["patients"], sep=";", dtype=int)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int)
    result = BOINDesign(
        target=float(row["target"]), extra_safe=row["extra_safe"] == "TRUE"
    ).select_mtd(n, y)
    ix = np.fromstring(row["doses"], sep=";", dtype=int) - 1
    interval = result.isotonic_interval[ix]
    if row["tite"] == "FALSE":
        interval = np.round(interval, 2)
    for field, actual in [
        ("means", result.isotonic_mean[ix]),
        ("lower", interval[:, 0]),
        ("upper", interval[:, 1]),
        ("overdose", result.report_overdose_probability[ix]),
    ]:
        np.testing.assert_allclose(
            actual, np.fromstring(row[field], sep=";"), rtol=1e-13, atol=1e-14
        )
