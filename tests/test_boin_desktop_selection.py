"""Active original selector with an independent six-cell isotonic oracle."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign
from mdanderson_stats.boin_desktop_selection import select_boin_desktop_mtd

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/selection.csv")
        .read_text()
        .splitlines()
    )
)


def values(text):
    return np.array([float(x) if x != "NA" else np.nan for x in text.split(";")]).reshape(2, 3)


@pytest.mark.parametrize("row", ROWS)
def test_original_selector_and_interval_report(row):
    assert not row["error"]
    design = BOINDesktopCombDesign(target=0.25, extra_safe=row["extra_safe"] == "TRUE")
    result = select_boin_desktop_mtd(
        design,
        values(row["patients"]),
        values(row["toxicities"]),
        contour=row["contour"] == "TRUE",
        profile="desktop_legacy",
    )
    expected = tuple(zip(*(iter([int(x) for x in row["selected"].split(";") if x]),) * 2))
    expected = tuple(d for d in expected if d != (99, 99))
    actual = (
        result.selection.contour
        if row["contour"] == "TRUE"
        else (() if result.selection.dose is None else (result.selection.dose,))
    )
    assert actual == expected
    np.testing.assert_allclose(
        result.reported_mean, values(row["mean"]), atol=2e-10, rtol=2e-10, equal_nan=True
    )
    np.testing.assert_allclose(result.lower, values(row["lower"]), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.upper, values(row["upper"]), atol=1e-12, equal_nan=True)


def test_legacy_unsafe_selection_is_explicit_and_not_default():
    n = np.array([[3, 0, 0], [0, 0, 60]])
    y = np.array([[0, 0, 0], [0, 0, 23]])
    design = BOINDesktopCombDesign(target=0.25)
    safe = select_boin_desktop_mtd(design, n, y)
    legacy = select_boin_desktop_mtd(design, n, y, profile="desktop_legacy")
    assert safe.selection.dose == (1, 1) and not safe.selected_eliminated
    assert legacy.selection.dose == (2, 3) and legacy.selected_eliminated
    assert not safe.reported_mean.flags.writeable
    assert select_boin_desktop_mtd(design, n, y, intervals=False).lower is None
