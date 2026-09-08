"""BP1CI fractional formulas, interface constraints and native endpoint checks."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import betainc, betaincc

from mdanderson_stats import binomial_interval, bp1ci, bp1ci_binomial_interval

CASES = json.loads((Path(__file__).parent / "fixtures/bp1ci_extended.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_extended_intervals(case):
    settings = {k: v for k, v in case.items() if k not in ("bounds", "atol", "source_issue")}
    result = bp1ci(**settings)
    if "source_issue" in case:
        assert abs(float(result.lower) - case["bounds"][0]) > case["atol"]
        assert_allclose(result.upper, case["bounds"][1], atol=case["atol"], rtol=0)
    else:
        assert_allclose([result.lower, result.upper], case["bounds"], atol=case["atol"], rtol=0)


def test_fractional_lower_bound_below_one_has_closed_form():
    # Beta(k,1) CDF is x**k when successes=trials; source's s=k-1 is invalid.
    k = np.array([0.125, 0.5, 0.75])
    lower, upper = bp1ci_binomial_interval(k, k, 0.8)
    assert_allclose(lower, 0.1 ** (1 / k), rtol=1e-14)
    assert_allclose(upper, 1)
    lo, hi = bp1ci_binomial_interval(0, 0.5, 0.8)
    assert lo == 0
    assert_allclose(hi, 1 - 0.1**2)


def test_fractional_tail_equations_broadcast_and_symmetry():
    k = np.array([0.5, 1.25, 7.75])[:, None]
    n = 10.5
    confidence = np.array([0.001, 0.95, 0.999999])
    lo, hi = bp1ci_binomial_interval(k, n, confidence)
    tail = (1 - confidence) / 2
    assert_allclose(betainc(k, n - k + 1, lo), np.broadcast_to(tail, lo.shape), rtol=1e-8)
    assert_allclose(betaincc(k + 1, n - k, hi), np.broadcast_to(tail, hi.shape), rtol=1e-8)
    rlo, rhi = bp1ci_binomial_interval(n - k, n, confidence)
    assert_allclose(lo, 1 - rhi, atol=2e-15)
    assert_allclose(hi, 1 - rlo, atol=2e-15)


def test_integer_agreement_and_no_information():
    k = np.arange(11)
    assert_allclose(bp1ci_binomial_interval(k, 10), binomial_interval(k, 10))
    assert_allclose(bp1ci_binomial_interval(0, 0), [0, 1])


def test_entry_modes_estimate_scaling_and_report():
    a = bp1ci([12, 0], [18, 30])
    b = bp1ci([12, 0], [30, 30], entry="trials")
    assert_allclose(a.lower, b.lower)
    assert_allclose(a.estimate, [0.4, 0])
    assert "Failures\tTrials" in a.report()
    assert len(a.report().splitlines()) == 4
    p = bp1ci(10000, distribution="poisson", confidence_percent=99, exposure=100)
    raw = bp1ci(10000, distribution="poisson", confidence_percent=99)
    assert_allclose([p.lower, p.upper], np.array([raw.lower, raw.upper]) / 100)
    assert p.estimate == 100 and "not the Garwood" in p.report()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"events": -1, "second": 1},
        {"events": 1, "second": None},
        {"events": 0, "second": 0},
        {"events": 3, "second": 2, "entry": "trials"},
        {"events": 1e10 + 1, "second": 1},
        {"events": 1, "second": 1, "confidence_percent": 0.09},
        {"events": 1, "second": 1, "confidence_percent": 99.99999},
        {"events": 1, "second": 1, "exposure": 2},
        {"events": 1, "distribution": "poisson", "second": 1},
        {"events": 0, "distribution": "poisson"},
        {"events": 1, "second": 1, "entry": "other"},
    ],
)
def test_invalid_interface_inputs(kwargs):
    with pytest.raises(ValueError):
        bp1ci(**kwargs)


@pytest.mark.parametrize("k,n", [(-1, 2), (3, 2), (1, np.inf), (np.nan, 2), (1, 2**53)])
def test_invalid_fractional_inputs(k, n):
    with pytest.raises(ValueError):
        bp1ci_binomial_interval(k, n)


def test_independent_decimal_tail_for_native_discrepancy():
    # B(5/4,15/4) = 77*pi*sqrt(2)/2048 from gamma recurrence/reflection.
    # Integrate the binomial series for (1-t)**(b-1) independently of SciPy.
    lower = float(bp1ci(1.25, 2.75, confidence_percent=0.1).lower)
    with localcontext() as context:
        context.prec = 60
        a, b = Decimal("1.25"), Decimal("3.75")
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
        normalizer = Decimal(77) * pi * Decimal(2).sqrt() / 2048
        x = Decimal(str(lower))
        term, total = Decimal(1), 1 / a
        for n in range(1, 200):
            term *= (Decimal(n) - b) * x / n
            total += term / (a + n)
        cdf = x**a * total / normalizer
        assert abs(cdf - Decimal(".4995")) < Decimal("1e-15")
