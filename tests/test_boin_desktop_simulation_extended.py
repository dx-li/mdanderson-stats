"""Full original wrappers including extra-safe and custom run-in bounds."""

import csv
from pathlib import Path

import pytest
from test_boin_desktop_simulation import test_complete_original_wrapper as check_wrapper

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/simulation-extended.csv")
        .read_text()
        .splitlines()
    )
)


@pytest.mark.parametrize("row", ROWS)
def test_extended_original_wrappers(row):
    assert row["error"] == ""
    check_wrapper(row)
