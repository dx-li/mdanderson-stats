"""ONESAMPLE interface/report behavior against native calculation fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import one_sample

CASES = json.loads((Path(__file__).parent / "fixtures/onesample.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_confidence_and_test_workflows_against_native(case):
    c = case
    distribution = c["distribution"]
    options = (
        {"second": c["failures"]} if distribution == "binomial" else {"exposure": c["exposure"]}
    )
    result = one_sample(distribution + "_confidence", c["events"], c["confidence"], **options)
    assert_allclose(result.estimate, c["estimate"])
    assert_allclose([result.lower, result.upper], c["bounds"], rtol=2e-8, atol=1e-12)
    if 1e-10 <= c["null"] <= (1 - 1e-10 if distribution == "binomial" else 1e9):
        result = one_sample(
            distribution + "_test", c["events"], c["null"], legacy_cutoffs=True, **options
        )
        assert_allclose(
            [result.p_less, result.p_greater], [c["p_less"], c["p_greater"]], rtol=2e-9, atol=1e-15
        )
        assert result.lower is None and result.upper is None
    if distribution == "binomial":
        total = c["events"] + c["failures"]
        alternate = one_sample(
            "binomial_confidence", c["events"], c["confidence"], second=total, entry="trials"
        )
        assert_allclose([alternate.lower, alternate.upper], c["bounds"], rtol=2e-8, atol=1e-12)


def test_report_labels_and_numeric_fields(tmp_path):
    cases = [
        one_sample("binomial_confidence", 12, 0.95, second=18),
        one_sample("binomial_test", 12, 0.3, second=30, entry="trials"),
        one_sample("poisson_confidence", 10, 0.95, exposure=5),
        one_sample("poisson_test", 10, 2, exposure=5),
    ]
    for result in cases:
        lines = result.report(digits=15).splitlines()
        record = dict(zip(lines[1].split("\t"), map(float, lines[2].split("\t")), strict=True))
        assert record["Events"] == float(result.events)
        assert_allclose(record["Estimate"], result.estimate, rtol=1e-14)
        if result.lower is not None:
            assert record["Confidence"] == 0.95
            assert_allclose(
                [record["Lower"], record["Upper"]], [result.lower, result.upper], rtol=1e-14
            )
        else:
            assert_allclose(
                [record["P (less)"], record["P (greater)"]],
                [result.p_less, result.p_greater],
                rtol=1e-14,
            )
        path = result.write_report(tmp_path / "report.tsv", digits=15)
        assert path.read_text() == result.report(digits=15)


def test_broadcasting_retains_every_case_in_report():
    r = one_sample("binomial_test", [[1], [2]], [0.1, 0.5, 0.9], second=10, entry="trials")
    assert r.p_less.shape == (2, 3)
    rows = r.report().splitlines()[2:-1]
    assert len(rows) == 6
    assert_allclose([float(row.split("\t")[0]) for row in rows], [1, 1, 1, 2, 2, 2])


def test_zero_poisson_and_input_limit_endpoints():
    for confidence in (1e-10, 1 - 1e-10):
        r = one_sample("poisson_confidence", 0, confidence, exposure=1e-10)
        assert r.estimate == 0 and r.lower == 0 and np.isfinite(r.upper)
    r = one_sample("binomial_confidence", 1e9, 0.95, second=1e9)
    assert r.trials == 2e9 and r.lower < 0.5 < r.upper
    assert "1e+09" in r.report()
    r = one_sample("poisson_test", 1, 1e-10, legacy_cutoffs=True)
    assert r.p_greater == 0 and "cutoffs enabled" in r.report()


@pytest.mark.parametrize(
    "calculation,events,parameter,kwargs",
    [
        ("binomial_confidence", 0, 0.95, {"second": 0}),
        ("binomial_test", 3, 0.5, {"second": 2, "entry": "trials"}),
        ("binomial_confidence", 1.5, 0.95, {"second": 2}),
        ("binomial_test", 1, 0.5, {}),
        ("binomial_test", 1, 0.5, {"second": 1, "exposure": 2}),
        ("binomial_test", 1, 0.5, {"second": 1e9 + 1}),
        ("poisson_test", 1e9 + 1, 1, {}),
        ("poisson_test", 1, 0, {}),
        ("poisson_test", 1, 1, {"exposure": 1e9 + 1}),
        ("poisson_confidence", 1, 0.95, {"exposure": 0}),
        ("poisson_confidence", 1, 1, {}),
        ("poisson_test", 1, 1, {"second": 1}),
        ("poisson_test", 1, 1, {"entry": "trials"}),
        ("poisson_confidence", 1, 0.95, {"legacy_cutoffs": True}),
        ("other", 1, 0.95, {}),
    ],
)
def test_invalid_inputs(calculation, events, parameter, kwargs):
    with pytest.raises(ValueError):
        one_sample(calculation, events, parameter, **kwargs)


def test_report_precision_validation_before_file_write(tmp_path):
    r = one_sample("poisson_confidence", 1, 0.95)
    path = tmp_path / "existing"
    path.write_text("keep")
    with pytest.raises(ValueError):
        r.write_report(path, digits=0)
    assert path.read_text() == "keep"
    with pytest.raises(FileNotFoundError):
        r.write_report(tmp_path / "missing" / "file")
