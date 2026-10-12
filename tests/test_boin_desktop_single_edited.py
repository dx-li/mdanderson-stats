"""Adverse edited histories: source agreement or explicit safety exclusion."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/single-edited.csv")
        .read_text()
        .splitlines()
    )
)


class Probe(np.random.Generator):
    def integers(self, *args, **kwargs):
        return 0


def analyze(row):
    n = np.fromstring(row["patients"], sep=";", dtype=int).reshape(2, 3)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int).reshape(2, 3)
    design = BOINDesktopCombDesign(
        target=0.25,
        extra_safe=row["extra_safe"] == "TRUE",
        run_in=row["runin"] == "TRUE",
        early_stop_patients=12,
    )
    result = design.desktop_next_dose(
        n, y, (int(row["current_a"]), int(row["current_b"])), rng=Probe(np.random.PCG64(1))
    )
    expected = (int(row["next_a"]), int(row["next_b"]))
    return result, None if expected == (99, 99) else expected


@pytest.mark.parametrize("row", ROWS)
def test_native_adverse_history_or_deliberate_safety_exclusion(row):
    result, expected = analyze(row)
    if result.next_dose != expected:
        assert expected is not None
        assert result.eliminated[expected[0] - 1, expected[1] - 1]
    if result.next_dose:
        assert not result.eliminated[result.next_dose[0] - 1, result.next_dose[1] - 1]


def test_reference_divergence_is_bounded_and_explicit():
    results = [analyze(row) for row in ROWS]
    assert sum(result.next_dose == expected for result, expected in results) == 383
    assert sum(result.next_dose != expected for result, expected in results) == 97
