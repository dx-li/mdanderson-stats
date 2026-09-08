"""KSB1CI native agreement and exhaustive Bernoulli-path validation."""

import itertools
import json
from decimal import Decimal, localcontext
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KStageBinomial, binomial_interval, binomial_test

CASES = json.loads((Path(__file__).parent / "fixtures/ksb1ci.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_ksb1ci(case):
    c = case
    design = KStageBinomial(np.cumsum(c["sizes"]), c["low"], c["high"])
    less, _ = design.tails(c["stage"], c["events"], c["probability"])
    assert_allclose(less, c["cdf"], rtol=3e-5, atol=2e-7)
    lower, upper = design.interval(c["stage"], c["events"], c["confidence"])
    assert_allclose([lower, upper], [c["lower"], c["upper"]], atol=1.1e-4, rtol=0)


@pytest.mark.parametrize("p", [0, 0.01, 0.3, 0.8, 1])
def test_exhaustive_paths(p):
    totals = [2, 4, 6]
    low, high = [0, -1], [-1, 4]
    design = KStageBinomial(totals, low, high)
    for stage in (1, 2, 3):
        for events in range(totals[stage - 1] + 1):
            less = greater = 0.0
            reached = False
            for sequence in itertools.product((0, 1), repeat=totals[stage - 1]):
                weight = p ** sum(sequence) * (1 - p) ** (len(sequence) - sum(sequence))
                for i in range(stage):
                    count = sum(sequence[: totals[i]])
                    if i == stage - 1:
                        reached |= count == events
                        less += weight * (count <= events)
                        greater += weight * (count >= events)
                    elif count <= low[i]:
                        less += weight
                        break
                    elif high[i] >= 0 and count >= high[i]:
                        greater += weight
                        break
            if reached:
                assert_allclose(design.tails(stage, events, p), [less, greater], atol=1e-14)
            else:
                with pytest.raises(ValueError, match="unreachable"):
                    design.tails(stage, events, p)


def test_no_stopping_is_clopper_pearson_and_inclusive_binomial_tails():
    design = KStageBinomial([5, 12, 20], [-1, -1], [-1, -1])
    k = np.arange(21)[:, None]
    levels = np.array([0.8, 0.95, 0.999])
    assert_allclose(design.interval(3, k, levels), binomial_interval(k, 20, levels), atol=2e-14)
    p = np.array([0, 0.3, 1])
    expected = binomial_test(k, 20, p)
    assert_allclose(design.tails(3, k, p), [expected.p_less, expected.p_greater], atol=1e-14)


def test_bounds_invert_tails_and_have_at_least_nominal_coverage():
    design = KStageBinomial([2, 4, 6], [0, 1], [2, 4])
    for p in [0.01, 0.2, 0.5, 0.8, 0.99]:
        coverage = 0.0
        for sequence in itertools.product((0, 1), repeat=6):
            weight = p ** sum(sequence) * (1 - p) ** (6 - sum(sequence))
            for stage, n in enumerate(design.cumulative_trials, 1):
                k = sum(sequence[:n])
                if stage == 3 or k <= design.low[stage - 1] or k >= design.high[stage - 1]:
                    lower, upper = design.interval(stage, k, 0.9)
                    coverage += weight * (lower <= p <= upper)
                    if lower > 0:
                        assert_allclose(design.tails(stage, k, lower)[1], 0.05, atol=1e-14)
                    if upper < 1:
                        assert_allclose(design.tails(stage, k, upper)[0], 0.05, atol=1e-14)
                    break
        assert coverage >= 0.9 - 1e-14


def test_log_evaluation_does_not_lose_representable_tails():
    design = KStageBinomial([200], [], [])
    with localcontext() as ctx:
        ctx.prec = 100
        p = Decimal("0.02")
        expected = float(
            sum(Decimal(comb(200, k)) * p**k * (1 - p) ** (200 - k) for k in range(190, 201))
        )
    actual = design.tails(1, 190, 0.02)[1]
    assert actual > 0
    assert_allclose(actual, expected, atol=0, rtol=1e-12)


@pytest.mark.parametrize(
    "totals,low,high",
    [
        ([], [], []),
        ([0], [], []),
        ([201], [], []),
        ([2, 2], [-1], [-1]),
        ([2.5], [], []),
        ([2, 4], [], []),
        ([2, 4], [-2], [-1]),
        ([2, 4], [1], [1]),
        ([2, 4], [2], [-1]),
        ([2, 4], [-1], [0]),
        ([2, 4], [-1], [3]),
        ([2, 4], [0.5], [2]),
    ],
)
def test_invalid_design(totals, low, high):
    with pytest.raises(ValueError):
        KStageBinomial(totals, low, high)


@pytest.mark.parametrize(
    "stage,events,p",
    [
        (0, 1, 0.5),
        (True, 1, 0.5),
        (1.5, 1, 0.5),
        (2, 1, 0.5),
        (1, 3, 0.5),
        (1, 1.5, 0.5),
        (1, 1, -1),
        (1, 1, np.nan),
    ],
)
def test_invalid_query(stage, events, p):
    with pytest.raises(ValueError):
        KStageBinomial([2], [], []).tails(stage, events, p)


@pytest.mark.parametrize("level", [0, 1, np.nan])
def test_invalid_confidence(level):
    with pytest.raises(ValueError):
        KStageBinomial([2], [], []).interval(1, 1, level)


def test_design_and_interval_report_roundtrip(tmp_path):
    design = KStageBinomial([14, 28, 42], [0, 1], [3, 4])
    path = design.write_report(tmp_path / "report.tsv", 3, [3, 5], digits=14)
    lines = path.read_text().splitlines()
    assert "2\t14\t28\t1\t4" in lines
    assert "3\t14\t42\t—\t—" in lines
    values = np.array([list(map(float, row.split("\t"))) for row in lines[-2:]])
    assert_allclose(values[:, :3], [[3, 3, 0.95], [3, 5, 0.95]])
    assert_allclose(values[:, 3:].T, design.interval(3, [3, 5]), atol=1e-14)
    design.write_report(path, 2, 6, 0.8)
    assert path.read_text() == design.report(2, 6, 0.8)
    with pytest.raises(ValueError):
        design.write_report(path, 2, 6, digits=0)
    assert path.read_text() == design.report(2, 6, 0.8)
    with pytest.raises(FileNotFoundError):
        design.write_report(tmp_path / "missing" / "report", 2, 6)
