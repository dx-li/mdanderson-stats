"""Source-regression checks for the independent MTADF reference fixture."""

import csv
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "mtadf-author-reference.csv"


def _rows() -> list[dict[str, str]]:
    with _FIXTURE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_unweighted_author_fit_and_rightmost_peak_tie() -> None:
    rows = _rows()
    unequal = [row for row in rows if row["label"] == "unequal_n"]
    assert [int(row["subjects"]) for row in unequal] == [3, 6, 3, 12, 3]
    assert [float(row["fitted_rate"]) for row in unequal] == pytest.approx(
        [0.10, 0.80, 0.525, 0.525, 0.25]
    )

    plateau = [row for row in rows if row["label"] == "flat_peak_tie"]
    assert [float(row["fitted_rate"]) for row in plateau] == pytest.approx([0.20, 0.80, 0.80, 0.20])
    assert {int(row["peak_rightmost"]) for row in plateau} == {3}


def test_simulator_uses_previous_admissibility_cap() -> None:
    row = next(row for row in _rows() if row["label"] == "toxic_first_cohort_lag")
    assert int(row["initial_cap"]) == 3
    assert int(row["fresh_cap"]) == 1
    assert int(row["author_next_dose"]) == 2
    assert int(row["df_isotonic_next_dose"]) == 1


def test_zero_efficacy_final_tie_can_select_untried_dose() -> None:
    row = next(row for row in _rows() if row["label"] == "zero_final_untried_tie")
    assert float(row["fitted_rate"]) == 0
    assert int(row["fresh_cap"]) == 4
    assert int(row["peak_rightmost"]) == int(row["author_next_dose"]) == 4
