"""Native decision-table comparisons and readable-report integration checks."""

import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import BetaMixture, MultiSession

CASES = json.loads((Path(__file__).parent / "fixtures/multi_report.json").read_text())["cases"]


def decision_rows(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Step |"))
    result = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        result.append([field.strip() for field in line.split("|")[1:-1]])
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["method"])
def test_native_pdisp_rows(case):
    session = MultiSession(" ".join(format(p, ".17e") for p in case["pvalues"]))
    session.run("multiple_testing", method=case["method"])
    rows = decision_rows(session.format_report(digits=12))
    assert len(rows) == len(case["rows"])
    for actual, native in zip(rows, case["rows"], strict=True):
        assert list(map(int, actual[1:3])) == native[:2]
        assert_allclose(list(map(float, actual[3:5])), native[2:4], rtol=1e-5, atol=1e-12)
        assert (actual[-1] == "*") == native[-1]


def test_rom_threshold_label_and_rejection_summary():
    s = MultiSession(".001 .002 .5 .9")
    result = s.run("multiple_testing", method="rom", alpha=0.05)
    text = s.format_report(digits=17)
    assert "Critical alpha" in text and "Adjusted P-value" not in text
    rows = decision_rows(text)
    assert_allclose([float(r[4]) for r in rows], result.critical_values[result.order])
    assert f"Rejected: {sum(result.reject)} of 4." in text


def test_entered_sequence_preserves_step_and_true_rank():
    s = MultiSession(".8 .1 .6 .2")
    result = s.run("beta_mixture_testing", model=BetaMixture(1), sequence="input")
    rows = decision_rows(s.format_report())
    assert [list(map(int, r[:3])) for r in rows] == [[1, 4, 1], [2, 1, 2], [3, 3, 3], [4, 2, 4]]
    assert_allclose([float(r[4]) for r in rows], result.scores)
    assert "Reciprocal-density score" in s.format_report()


def test_reports_include_model_components_candidates_and_failures(tmp_path):
    s = MultiSession(".001 .01 .02 .1 .2 .3 .4 .5 .6 .7 .8 .9", seed=1)
    s.run("select_beta_mixture", max_components=1, algorithm="em")
    s.run("nonparametric_testing", null_estimate=5)
    s.run("schweder_bootstrap", samples=2)
    s.change_data(".1 .1 .1 .1", source="bad | <data>\nnext")
    with pytest.raises(ValueError):
        s.run("schweder_fit")
    text = s.format_report()
    assert "Component | Proportion | Beta a | Beta b" in text
    assert "Result.candidates[1]" in text
    assert "Result diagnostics.density" in text
    assert "Result.estimates" in text
    assert "**FAILED**" in text
    assert r"bad \| &lt;data&gt;<br>next" in text
    path = s.write_text_report(tmp_path / "report.md")
    assert path.read_text() == text
    with pytest.raises(FileNotFoundError):
        s.write_text_report(tmp_path / "missing" / "report.md")


def test_nonfinite_values_are_readable_and_not_dropped():
    s = MultiSession("0 .2 .3 1")
    s.run("beta_mixture_testing", model=BetaMixture(0, [1], [0.5], [2]))
    rows = decision_rows(s.format_report())
    assert rows[0][5] == "inf" and rows[-1][5] == "-inf"


@pytest.mark.parametrize("digits", [0, 18, True, 1.2])
def test_invalid_precision(digits):
    with pytest.raises(ValueError):
        MultiSession(".1 .2 .3 .4").format_report(digits=digits)
