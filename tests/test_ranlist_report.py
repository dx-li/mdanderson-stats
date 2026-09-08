import json
import re
from pathlib import Path

import pytest

from mdanderson_stats import (
    RanlistSession,
    RanlistSpecification,
    ranlist_report,
    ranlist_summary,
    read_ranlist_parameters,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranlist_reports.json").read_text())


def report_pages(text):
    pages = []
    for page in text.split("\f")[1:]:
        header = re.search(r"STRATUM (\d+).*PAGE (\d+)", page)
        assert header is not None
        rows = re.findall(r"^\s*(\d+)\s+(\d+)\s+[^\n]*\.{29}", page, re.MULTILINE)
        pages.append(
            {
                "stratum": int(header[1]),
                "page": int(header[2]),
                "patients": [int(row[0]) for row in rows],
                "treatments": [int(row[1]) for row in rows],
            }
        )
    return pages


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_report_rows_and_page_boundaries(case):
    state = read_ranlist_parameters(case["parameters"])
    actual = ranlist_report(state, case["requested"])
    assert report_pages(actual) == case["pages"]
    assert state.current_patients == (3, 0)
    for title in state.specification.title:
        assert title in actual
    for label in state.specification.treatments:
        assert label in actual


def test_modern_report_summarizes_choices_and_preserves_names():
    spec = RanlistSpecification(
        (2, 3),
        restricted=True,
        balance=(1, 4),
        strata=("S" * 30, "T" * 30),
        treatments=("A" * 30, "B" * 30),
        title=("X" * 80,),
        phrase="descriptive phrase",
    )
    state = RanlistSession(spec, (5, 7))
    text = ranlist_report(state, [0, 30])
    assert "X" * 80 in text
    assert "S" * 30 in text
    assert "A" * 30 in text
    assert "Seeds: 1234567890, 123456789" in text
    assert "descriptive phrase" in text
    assert "Multiplier redrawn at each refill" in text
    assert "Listed patients per stratum: 0, 30" in text
    pages = report_pages(text)
    assert [len(page["patients"]) for page in pages] == [26, 4]
    assert all(page["stratum"] == 2 for page in pages)
    expected = spec.allocate(list(range(1, 31)), strata=2).treatments.tolist()
    assert [value for page in pages for value in page["treatments"]] == expected
    assert state.current_patients == (5, 7)


def test_summary_does_not_generate_unenrolled_or_historical_blocks():
    state = RanlistSession(RanlistSpecification((1, 1), restricted=True, max_blocks=1), (100000,))
    text = ranlist_summary(state)
    assert "100000" in text
    assert "Balance multiplier: 1" in text
    assert "\f" not in text
    with pytest.raises(ValueError, match="max_blocks"):
        ranlist_report(state)


def test_empty_report_keeps_parameters_and_counters():
    state = RanlistSession(RanlistSpecification((0.123456789, 2), strata=("", "")))
    text = ranlist_report(state)
    assert "Relative weight" in text
    assert "0.123456789" in text
    assert "Balance multiplier" not in text
    assert "Listed patients per stratum: 0, 0" in text
    assert report_pages(text) == []


@pytest.mark.parametrize("patients", [-1, 1.5, [1, 2], [[1]], float("nan")])
def test_invalid_counts(patients):
    with pytest.raises(ValueError):
        ranlist_report(RanlistSession(RanlistSpecification((1,))), patients)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_row_limit(limit):
    with pytest.raises(ValueError, match="max_rows"):
        ranlist_report(RanlistSession(RanlistSpecification((1,))), max_rows=limit)


def test_total_row_limit_is_checked_before_allocation():
    state = RanlistSession(RanlistSpecification((1,), strata=("", "")))
    with pytest.raises(ValueError, match="max_rows"):
        ranlist_report(state, 2**52)
    with pytest.raises(ValueError, match="max_rows"):
        ranlist_report(state, [2, 3], max_rows=4)
    assert (
        sum(len(p["patients"]) for p in report_pages(ranlist_report(state, [2, 3], max_rows=5)))
        == 5
    )
