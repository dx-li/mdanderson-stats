"""Native desktop contour references, including subtrial transitions and caps."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_contour import desktop_contour_decision
from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/contour-conduct.csv")
        .read_text()
        .splitlines()
    )
)
ROWS = [r for r in ROWS if r["runin"] == "FALSE" or float(r["target"]) == 0.25]


@pytest.mark.parametrize("row", ROWS)
def test_original_desktop_contour(row):
    assert not row["error"]
    design = BOINDesktopCombDesign(
        target=float(row["target"]),
        toxic_probability=float(row["toxic_probability"]),
        extra_safe=row["extra_safe"] == "TRUE",
        run_in=row["runin"] == "TRUE",
        early_stop_patients=12,
    )
    n = np.fromstring(row["patients"], sep=";", dtype=int).reshape(2, 3)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int).reshape(2, 3)
    actual = desktop_contour_decision(
        design,
        n,
        y,
        (int(row["current_a"]), int(row["current_b"])),
        [12, 6],
        compatibility_profile="desktop_legacy",
    )
    status = int(row["managed_status"])
    if status == 0:
        assert actual.next_dose == (int(row["next_a"]), int(row["next_b"]))
    else:
        assert actual.next_dose is None
        assert (
            actual.action
            == {2: "stop_safety", 3: "complete", 4: "error", 5: "stop_precision"}[status]
        )
