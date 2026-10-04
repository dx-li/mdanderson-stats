"""User-facing static-report behavior for TTEConduct boundaries."""

import csv
from pathlib import Path

import pytest

from mdanderson_stats.tteconduct import tteconduct_design
from mdanderson_stats.tteconduct_report import TTEConductReport, tteconduct_report


def _design(*, cutoff: float = 0.03, cap: float = 4800.0):
    return tteconduct_design(60, 295, 3, 10, 1, cutoff, 40, max_total_time=cap)


def test_report_contains_source_inputs_and_boundary_fixture_values(tmp_path):
    report = tteconduct_report(_design(), time_unit="months", events=[1, 2, 3])
    reference = Path(__file__).parent / "fixtures" / "tteconduct-reference.csv"
    with reference.open(newline="", encoding="utf-8") as stream:
        expected = {int(row["events"]): row for row in csv.DictReader(stream)}

    assert [row.events for row in report.boundaries.boundaries] == [1, 2, 3]
    assert report.boundaries.boundaries[2].minimum_total_time == pytest.approx(
        float(expected[3]["minimum_total_time"]), abs=2e-8
    )
    html = report.to_html()
    for field in (
        "Standard prior shape αS",
        "Standard prior scale βS",
        "Experimental prior shape αE",
        "Experimental prior scale βE",
        "Required additive improvement δ",
        "Futility cutoff c",
        "Maximum patients m",
        "Boundary-search cap",
        "Absolute quadrature tolerance",
    ):
        assert field in html
    expected_boundary = format(report.boundaries.boundaries[2].minimum_total_time, ".17g")
    assert f"{expected_boundary} months" in html
    assert "P(μE &gt; μS + δ | data) &lt; c" in html
    output = report.write_html(tmp_path / "tteconduct.html")
    assert output.read_text(encoding="utf-8") == html


def test_report_explains_zero_boundary_and_search_cap():
    zero = tteconduct_report(_design(), time_unit="months", events=1)
    assert "Continuation criterion met at zero exposure" in zero.to_html()

    capped = tteconduct_report(_design(cutoff=0.999999, cap=1), time_unit="months", events=3)
    html = capped.to_html()
    assert "Not resolved within search cap" in html
    assert "not proof continuation is impossible" in html
    assert "Not resolved" in html


def test_report_escapes_labels_and_renders_before_replacing_file(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        tteconduct_report(_design(), time_unit="")
    with pytest.raises(ValueError):
        tteconduct_report(_design(), time_unit="months", title="bad\ntitle")

    report = tteconduct_report(
        _design(), time_unit="months <&>", title="Report <script>alert(1)</script>", events=1
    )
    html = report.to_html()
    assert "Report &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "months &lt;&amp;&gt;" in html
    assert "<script>alert(1)</script>" not in html

    destination = tmp_path / "report.html"
    destination.write_text("previous report", encoding="utf-8")

    def fail_render(self):
        raise ValueError("render failed")

    monkeypatch.setattr(TTEConductReport, "to_html", fail_render)
    with pytest.raises(ValueError, match="render failed"):
        report.write_html(destination)
    assert destination.read_text(encoding="utf-8") == "previous report"
