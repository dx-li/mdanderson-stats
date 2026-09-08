"""CUMINC pointwise intervals, step summaries and persisted numerical reports."""

import io
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import cumulative_incidence

CASES = json.loads((Path(__file__).parent / "fixtures/cuminc.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("confidence", [0.8, 0.95])
def test_archived_summary_formula_on_native_curve_corners(case, confidence):
    curve = cumulative_incidence(case["time"], case["event"])
    table = curve.summary(confidence=confidence)
    native = np.asarray(case["corners"])
    se = np.sqrt(native[:, 2])
    # Independent standard-library quantile applied to native Fortran variances.
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    expected = np.column_stack(
        [
            native[:, :2],
            se,
            np.maximum(0, native[:, 1] - z * se),
            np.minimum(1, native[:, 1] + z * se),
        ]
    )
    assert_allclose(table.rows, expected, rtol=2e-11, atol=3e-15)


def test_explicit_times_use_right_limits_and_preserve_order_and_duplicates():
    curve = cumulative_incidence([0, 1, 2, 3], [1, 1, 2, 0])
    times = np.array([3, 0, 1, 1, 20])
    table = curve.summary(times)
    assert_array_equal(table.rows[:, 0], times)
    assert_allclose(table.rows[:, 1], [0.5, 0.25, 0.5, 0.5, 0.5])
    assert curve.summary().rows[0, 1] == 0
    assert curve.summary(0).rows.shape == (1, 5)
    times[:] = 9
    assert table.rows[0, 0] == 3
    with pytest.raises(ValueError):
        table.rows[0, 0] = 10


def test_zero_curve_and_highest_representable_confidence_are_finite():
    level = np.nextafter(1.0, 0.0)
    curve = cumulative_incidence([1, 2, 3], [0, 2, 0])
    assert_array_equal(curve.summary(confidence=level).rows[:, 1:], 0)
    curve = cumulative_incidence([1], [1])
    assert_array_equal(curve.summary(1, confidence=level).rows, [[1, 1, 1, 0, 1]])


def test_confidence_intervals_expand_with_confidence():
    curve = cumulative_incidence(range(100), [1] * 30 + [2] * 30 + [0] * 40)
    narrow, wide = curve.summary(confidence=0.5), curve.summary(confidence=0.99)
    assert np.all(wide.rows[:, 3] <= narrow.rows[:, 3])
    assert np.all(wide.rows[:, 4] >= narrow.rows[:, 4])
    assert_allclose(wide.rows[:, :3], narrow.rows[:, :3])


def test_report_labels_precision_and_file_round_trip(tmp_path):
    summary = cumulative_incidence([1, 2, 3], [1, 2, 0]).summary(confidence=0.8)
    text = summary.report(digits=17)
    assert text.splitlines()[0] == "time\tincidence\tstd.err\tlower 80% CI\tupper 80% CI"
    assert_allclose(np.loadtxt(io.StringIO(text), skiprows=1), summary.rows, rtol=0, atol=0)
    destination = tmp_path / "summary.tsv"
    destination.write_text("replace this")
    summary.write_report(destination, digits=17)
    assert destination.read_text() == text
    assert summary.report(digits=3) != text
    with pytest.raises(FileNotFoundError):
        summary.write_report(tmp_path / "absent" / "report.tsv")


def test_empty_requested_times_give_header_only_report():
    table = cumulative_incidence([1], [1]).summary([])
    assert table.rows.shape == (0, 5)
    assert len(table.report().splitlines()) == 1


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan, np.inf, [0.95]])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        cumulative_incidence([1], [1]).summary(confidence=confidence)


@pytest.mark.parametrize("times", [-1, [np.inf], [[1]], [np.nan]])
def test_invalid_times(times):
    with pytest.raises(ValueError):
        cumulative_incidence([1], [1]).summary(times)


@pytest.mark.parametrize("digits", [0, 18, 1.5, True])
def test_invalid_report_precision(digits):
    with pytest.raises(ValueError):
        cumulative_incidence([1], [1]).summary().report(digits=digits)
