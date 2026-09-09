"""Independent gamma integrals and density factors for the unchanged F95 audit."""

import json
import math
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path

import pytest
from test_cdflib_error_exponential_reference import decimal_pi
from test_cdflib_gamma_support_reference import EULER, positive_log_gamma

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_incomplete_gamma.json").read_text())


def log_gamma(a):
    if a < Decimal("1e-30"):
        return -a.ln() - EULER * a + decimal_pi() ** 2 * a * a / 12
    return positive_log_gamma(a)


@cache
def independent(a, x):
    with localcontext() as ctx:
        ctx.prec = 800
        a, x = Decimal.from_float(a), Decimal.from_float(x)
        if a == 0:
            return 0.0, 1.0, 0.0
        if x == 0:
            return 0.0, 0.0, 1.0
        log_r = a * x.ln() - x - log_gamma(a)
        r = log_r.exp() if log_r > -10000 else Decimal(0)
        if a > 10000:
            # Only equal-shape and far-left large-shape cases are recorded.
            # At the mean, the first correction to 1/2 is O(a**(-1/2));
            # for a>=1e100 it is below float64 resolution.
            assert a >= Decimal("1e100") and (x == a or x == 1)
            return float(r), (0.5 if x == a else 0.0), (0.5 if x == a else 1.0)
        if a == a.to_integral_value():
            term = total = Decimal(1)
            for k in range(1, int(a)):
                term *= x / k
                total += term
            q = (-x).exp() * total
            return float(r), float(1 - q), float(q)
        if x >= 100:
            # Integration by parts gives the alternating upper-tail expansion.
            # After 80 terms the remaining integral has negative exponent;
            # the first omitted term bounds the error for these a<=20 cases.
            assert a <= 20
            term = total = Decimal(1)
            for k in range(1, 81):
                term *= (a - k) / x
                total += term
            omitted = abs(term * (a - 81) / x)
            assert omitted < abs(total) * Decimal("1e-35")
            q = r * total / x
            return float(r), float(1 - q), float(q)
        # Positive lower-gamma power series. Extra precision preserves tiny
        # complements when a is subnormal; local log Gamma avoids oracle error.
        term = total = Decimal(1)
        for k in range(1, 10000):
            term *= x / (a + k)
            total += term
            if term < total * Decimal("1e-750"):
                break
        else:
            raise AssertionError("independent lower-gamma series did not converge")
        p = r * total / a
        return float(r), float(p), float(1 - p)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_incomplete_gamma(case):
    assert case["execution_outcome"] == "completed"
    a, x, mode = case["a"], case["x"], case["mode"]
    if a < 0 or x < 0 or (a == 0 and x == 0):
        if mode == 2:
            assert case["p"] == 2 and case["q"] == -999
        elif mode == 3:
            assert case["p"] == 0 and case["q"] == 1
        else:
            assert mode == 1 and a == -0.5
            assert case["r"] < 0
        return
    r, p, q = independent(a, x)
    if mode == 1:
        assert case["r"] == pytest.approx(r, rel=5e-13, abs=5e-324)
        return
    if mode == 2 and a >= 1e100 and a == x:
        assert case["p"] == 2 and case["q"] == -999
        assert p == q == 0.5
        return
    if mode == 2 and a in (20, 100) and x != a and case["ind"] == 0:
        # The off-center large-shape branch applies a positive Stirling
        # correction where the gamma density needs its negative.
        smaller = "p" if p < q else "q"
        expected = min(p, q)
        relative_error = case[smaller] / expected - 1
        lo, hi = (0.0083, 0.0084) if a == 20 else (0.0016, 0.0017)
        assert lo < relative_error < hi
        assert case[smaller] == pytest.approx(expected * math.exp(1 / (6 * a)), rel=5e-13)
        assert case["p"] + case["q"] == 1
        return
    if mode == 2 and a == x == 20 and case["ind"] != 0:
        assert case["p"] < 0.5 < p
        assert 0.0594 < p - case["p"] < 0.0596
        assert case["p"] + case["q"] == 1
        return
    if a > 0 and x > 0 and a * x == 0:
        assert case["p"] == int(x > a) and case["q"] == int(x <= a)
        assert (p if x <= a else q) > 0
        return
    if mode == 3 and case["eps"] <= 0:
        assert case["p"] == case["q"] == "nan"
        return
    if mode == 3 and x >= 1.1:
        q *= case["factor"]
        p = 1 - q
    tolerance = 5e-13
    if mode == 2 and case["ind"] != 0:
        tolerance = 1e-6 if case["ind"] == 1 else 1e-3
    if mode == 3:
        tolerance = max(tolerance, case["eps"])
    assert case["p"] == pytest.approx(p, rel=tolerance, abs=5e-324)
    assert case["q"] == pytest.approx(q, rel=tolerance, abs=5e-324)


@pytest.mark.parametrize("x", [0.1, 1.1, 10.0, 100.0, 744.0])
def test_oracle_exponential_identity(x):
    r, p, q = independent(1.0, x)
    assert q == pytest.approx(math.exp(-x), rel=2e-15, abs=5e-324)
    expected_r = x * math.exp(-x) if x < 700 else math.exp(math.log(x) - x)
    assert r == pytest.approx(expected_r, rel=2e-15, abs=5e-324)
    assert p + q == 1


@pytest.mark.parametrize("x", [0.1, 1.1, 10.0, 100.0])
def test_oracle_half_shape_identity(x):
    _, p, q = independent(0.5, x)
    assert p == pytest.approx(math.erf(math.sqrt(x)), rel=5e-15)
    assert q == pytest.approx(math.erfc(math.sqrt(x)), rel=5e-15)


@pytest.mark.parametrize("x", [0.1, 1.1])
def test_tiny_shape_upper_tail_against_exponential_integral(x):
    with localcontext() as ctx:
        ctx.prec = 100
        z = Decimal.from_float(x)
        term = total = -z
        for k in range(2, 200):
            term *= -z / k
            total += term / k
        e1 = -EULER - z.ln() - total
        expected = float(Decimal.from_float(1e-100) * e1)
    assert independent(1e-100, x)[2] == pytest.approx(expected, rel=1e-15, abs=0)
