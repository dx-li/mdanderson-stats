"""Adverse edited histories must not silently assign globally eliminated doses."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_contour import desktop_contour_decision
from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/edited-history.csv")
        .read_text()
        .splitlines()
    )
)


@pytest.mark.parametrize("row", ROWS)
def test_original_edited_history_and_safe_assignment(row):
    n = np.fromstring(row["patients"], sep=";", dtype=int).reshape(2, 3)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int).reshape(2, 3)
    design = BOINDesktopCombDesign(
        target=0.25, early_stop_patients=12, extra_safe=row["extra_safe"] == "TRUE"
    )
    current = (int(row["current_a"]), int(row["current_b"]))
    legacy = desktop_contour_decision(
        design, n, y, current, [24, 24], compatibility_profile="desktop_legacy"
    )
    statuses = {"complete": 3, "stop_safety": 2, "stop_precision": 5, "error": 4}
    assert row["error"] == ""
    assert statuses.get(legacy.action, 0) == int(row["managed_status"])
    if legacy.next_dose is not None:
        assert legacy.next_dose == (int(row["next_a"]), int(row["next_b"]))
    safe = desktop_contour_decision(design, n, y, current, [24, 24])
    if safe.next_dose is not None:
        assert not safe.eliminated[safe.next_dose[0] - 1, safe.next_dose[1] - 1]
