"""Original desktop interactive cases, separate from its simulation wrappers."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/single-conduct.csv").open()
    )
)
ROWS = [r for r in ROWS if r["runin"] == "FALSE" or float(r["target"]) == 0.25]


@pytest.mark.parametrize("row", ROWS)
def test_native_single_conduct(row):
    design = BOINDesktopCombDesign(
        target=float(row["target"]),
        toxic_probability=float(row["toxic_probability"]),
        run_in=row["runin"] == "TRUE",
        extra_safe=row["extra_safe"] == "TRUE",
        early_stop_patients=12,
    )
    n = np.fromstring(row["patients"], sep=";", dtype=int).reshape(2, 3)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int).reshape(2, 3)
    dose = (int(row["current_a"]), int(row["current_b"]))
    actual = design.desktop_next_dose(n, y, dose, rng=1)
    expected = (int(row["next_a"]), int(row["next_b"]))
    assert actual.next_dose == (None if expected == (99, 99) else expected)


def test_desktop_convergence_does_not_unconditionally_stop_at_threshold():
    n = np.array([[12, 0, 0], [0, 0, 0]])
    y = np.zeros_like(n)
    result = BOINDesktopCombDesign(target=0.25, early_stop_patients=12).desktop_next_dose(
        n, y, (1, 1), rng=1
    )
    assert result.action == "escalate" and result.next_dose in ((2, 1), (1, 2))


def test_runin_gate_and_boundaries():
    with pytest.raises(ValueError, match="target"):
        BOINDesktopCombDesign(target=0.3, run_in=True)
    with pytest.raises(ValueError, match="boolean"):
        BOINDesktopCombDesign(run_in=1)
    design = BOINDesktopCombDesign(target=0.25, toxic_probability=0.5, run_in=True)
    assert design.boundary_table(6).deescalate_min[2] == 2
    assert design.boundary_table(6).deescalate_min[5] == 3
