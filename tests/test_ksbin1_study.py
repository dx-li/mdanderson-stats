"""Comparison/report/revision workflows with independent and native references."""

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import ksbin1_study

NATIVE = json.loads((Path(__file__).parent / "fixtures/ksbin1.json").read_text())["cases"]


@pytest.mark.parametrize("case", NATIVE)
def test_native_stage_summaries_and_independent_comparison(case):
    c = case
    p = c["probability"]
    less = c["alternative"] == "less"
    p0, pa = ((1 + p) / 2, p) if less else (p / 2, p)
    r = ksbin1_study(
        c["totals"],
        c["critical"],
        c["quit"],
        p0,
        pa,
        c["critical"][-1],
        alternative=c["alternative"],
    )
    for name in ["rejection", "quitting", "continuation"]:
        assert_allclose(getattr(r.characteristics, name)[1], c[name], rtol=5e-13, atol=1e-15)
    assert_allclose(r.characteristics.expected_sample_size[1], c["expected"][0], rtol=5e-13)
    assert_allclose(r.characteristics.expected_given_rejection[1], c["expected"][1], rtol=5e-13)
    n, cutoff = c["totals"][-1], c["critical"][-1]
    region = range(cutoff + 1) if less else range(cutoff, n + 1)
    expected = [sum(comb(n, k) * q**k * (1 - q) ** (n - k) for k in region) for q in [p0, pa]]
    assert_allclose([r.single_stage_significance, r.single_stage_power], expected, rtol=1e-13)
    # Parse the numeric report, verifying its actual values rather than just labels.
    lines = r.report(digits=17).splitlines()
    rows = [line.split("\t") for line in lines]
    stage_start = lines.index("Stage decisions") + 2
    for i in range(len(c["totals"])):
        values = list(map(float, rows[stage_start + i]))
        assert_allclose([values[5], values[7]], [c["rejection"][i], c["quitting"][i]], atol=1e-14)
    cumulative = lines.index("Cumulative decisions") + 2
    for i in range(len(c["totals"])):
        values = list(map(float, rows[cumulative + i]))
        assert_allclose(
            values[4:],
            [sum(c["rejection"][: i + 1]), sum(c["quitting"][: i + 1]), c["continuation"][i]],
            atol=1e-14,
        )
    expectation = lines.index("Expected observations") + 3
    values = list(map(float, rows[expectation][1:]))
    assert_allclose(
        values,
        [
            c["expected"][1],
            c["expected"][0],
            c["expected"][0] - c["expected"][1],
            n - c["expected"][0],
        ],
        rtol=1e-12,
        atol=1e-12,  # Subtracting nearly equal expected sample sizes amplifies relative error.
    )


def example():
    return ksbin1_study([14, 28, 42], [0, 1, 3], [3, 4], 0.2, 0.06, 4)


def test_design_export_and_file_output(tmp_path):
    r = example()
    expected = "3\n14 3 0\n14 4 1\n14 -1 3\n"
    assert r.design_text() == expected
    path = tmp_path / "design.txt"
    path.write_text("old")
    assert r.write_design(path) == path
    assert path.read_text() == expected
    report = tmp_path / "study.tsv"
    assert r.write_report(report, include_tables=True) == report
    assert report.read_text() == r.report(include_tables=True)
    assert report.read_text().count("Boundary assistance: stage") == 3
    # Greater-tail exports retain quit/critical column order, not low/high order.
    greater = r.revise(
        null_probability=0.8,
        alternative_probability=0.94,
        critical=[14, 27, 39],
        quit=[11, 24],
        single_stage_critical=38,
        alternative="greater",
    )
    assert greater.design_text() == "3\n14 11 14\n14 24 27\n14 -1 39\n"
    assert_allclose(greater.power, r.power, atol=1e-14)


def test_revision_recomputes_and_preserves_original():
    r = example()
    before = r.report(digits=17)
    changed = r.revise(
        null_probability=[0.2, 0.3],
        alternative_probability=0.08,
        critical=[-1, 1, 4],
        single_stage_critical=5,
    )
    direct = ksbin1_study([14, 28, 42], [-1, 1, 4], [3, 4], [0.2, 0.3], 0.08, 5)
    assert_allclose(changed.power, direct.power)
    assert_allclose(changed.significance, direct.significance)
    assert_allclose(changed.boundary_table(2).power_loss, direct.boundary_table(2).power_loss)
    assert r.report(digits=17) == before
    new = r.revise(cumulative_trials=[10], critical=[1], quit=[], single_stage_critical=1)
    assert_allclose(new.power, new.single_stage_power)
    assert_allclose(new.significance, new.single_stage_significance)
    assert_allclose(new.expected_sample_savings, 0, atol=1e-13)


def test_broadcast_report_cases_and_undefined_expectation():
    r = ksbin1_study([2, 4], [-1, -1], [-1], [[0.5], [1]], [0, 0.2], 1)
    assert r.characteristics.rejection.shape == (2, 2, 2, 2)
    assert r.power.shape == (2, 2)
    assert np.all(r.power == 0)
    assert np.all(np.isnan(r.characteristics.expected_given_rejection))
    report = r.report(include_tables=True)
    assert sum(line.startswith("Case ") for line in report.splitlines()) == 4
    assert "Ha\tnan\t4\tnan" in report


@pytest.mark.parametrize("digits", [0, 18, True, 2.5])
def test_invalid_report_precision(digits, tmp_path):
    path = tmp_path / "report"
    path.write_text("preserve")
    with pytest.raises(ValueError):
        example().write_report(path, digits=digits)
    assert path.read_text() == "preserve"


def test_errors_propagate(tmp_path):
    r = example()
    with pytest.raises(ValueError):
        r.report(include_tables="yes")
    with pytest.raises(ValueError):
        r.revise(null_probability=0.01)
    with pytest.raises(ValueError):
        r.revise(cumulative_trials=[10])
    with pytest.raises(OSError):
        r.write_report(tmp_path / "absent" / "report")
    with pytest.raises(OSError):
        r.write_design(tmp_path / "absent" / "design")
