"""Reviewable design exports preserve numerical optimization results."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import SingleOptimizedDesign, single_optimize_design


@pytest.mark.parametrize("two_samples", [False, True])
def test_optimized_report_reconstructs_design(two_samples):
    doses = ([-1, 1], [-1, 1]) if two_samples else [-1, 1]
    r = single_optimize_design(
        doses,
        [0, 1, 1] if two_samples else [0, 1],
        [-5, 5],
        comparison="slope" if two_samples else None,
        criterion="slope",
    )
    text = r.report(digits=17)
    lines = text.splitlines()
    end = lines.index("")
    records = [line.split("\t") for line in lines[2:end]]
    assert_allclose(sum(float(row[3]) for row in records), 100)
    xgroups = r.doses if two_samples else (r.doses,)
    ngroups = r.subjects if two_samples else (r.subjects,)
    for group, (x, n) in enumerate(zip(xgroups, ngroups, strict=True), start=1):
        subset = [row for row in records if int(row[0]) == group]
        assert [int(row[1]) for row in subset] == list(range(1, len(x) + 1))
        assert_allclose([float(row[2]) for row in subset], x, rtol=0, atol=0)
        assert_allclose([float(row[3]) for row in subset], n, rtol=0, atol=0)
    metrics = dict(line.split("\t") for line in lines[lines.index("Metric\tValue") + 1 :])
    assert float(metrics["Final criterion"]) == r.value
    assert float(metrics["Initial criterion"]) == r.initial_value
    assert float(metrics["Stationarity diagnostic"]) == r.stationarity
    assert int(metrics["Iterations"]) == r.iterations


def test_zero_allocations_and_original_order_are_retained(tmp_path):
    r = SingleOptimizedDesign(
        np.array([2.0, 0.0, -2.0]), np.array([50.0, 0.0, 50.0]), 0.2, 0.4, 1e-8, 12
    )
    report = r.report()
    assert "1\t1\t2\t50\n1\t2\t0\t0\n1\t3\t-2\t50" in report
    path = tmp_path / "design.tsv"
    path.write_text("old report")
    assert r.write_report(path) == path
    assert path.read_text() == report
    with pytest.raises(IsADirectoryError):
        r.write_report(tmp_path)


@pytest.mark.parametrize("digits", [0, 18, True, 3.5])
def test_invalid_precision(digits):
    r = SingleOptimizedDesign(np.array([-1.0, 1.0]), np.array([50.0, 50.0]), 0.2, 0.3, 0.0, 1)
    with pytest.raises(ValueError):
        r.report(digits=digits)
