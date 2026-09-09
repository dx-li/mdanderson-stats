"""Independent beta integrals for the three native series helpers."""

import json
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path

import pytest
from test_cdflib_beta_factors_reference import log_beta

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_beta_series.json").read_text())


@cache
def independent(mode, a, b, x):
    with localcontext() as ctx:
        ctx.prec = 800
        a, b, x = (v if isinstance(v, Decimal) else Decimal.from_float(v) for v in (a, b, x))
        if x == 0:
            return 1.0 if mode == 1 else 0.0
        if x == 1:
            return 0.0 if mode == 1 else 1.0
        if a == 1:
            upper = ((1 - x).ln() * b).exp()
            return float(upper if mode == 1 else 1 - upper)
        if b == 1:
            lower = (x.ln() * a).exp()
            return float(1 - lower if mode == 1 else lower)
        assert x <= Decimal(".5") and (b <= 1 or b * x <= 100)
        # Integrate the binomial expansion of (1-t)**(b-1) term by term.
        term = total = Decimal(1)
        for k in range(1, 10000):
            term *= (k - b) * x / k
            add = term * a / (a + k)
            total += add
            if abs(add) < abs(total) * Decimal("1e-750"):
                break
        else:
            raise AssertionError("independent beta integral failed to converge")
        exponent = a * x.ln() - a.ln() - log_beta(a, b)
        lower = exponent.exp() * total
        return float(1 - lower if mode == 1 else lower)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_series_against_independent_beta_integrals(case):
    mode, a, b, x, eps = (case[k] for k in ("mode", "a", "b", "x", "eps"))
    expected = independent(mode, a, b, x)
    if case["execution_outcome"] == "timeout":
        assert eps <= 0 or (mode == 3 and a == 1 and b == 0.5 and x > 0.99)
        assert 0 <= expected <= 1
        return
    assert case["execution_outcome"] == "completed"
    result = case["result"]
    if mode == 1 and x == 0:
        assert result == "inf" and expected == 1
    elif mode == 1 and b == 1e-309:
        assert result == "inf" and expected == pytest.approx(a / b, rel=5e-13)
    elif mode == 2 and b == 1e-15:
        assert result == 0 and expected == 1e-323
    elif mode == 2 and x == 0 and a <= 1e-3 * eps:
        assert result == b / a and expected == 0
    elif mode == 2 and eps == 1e-3:
        assert 5e-4 < 1 - result / expected < 6e-4
        assert abs(result - expected) <= eps * expected
    else:
        assert result == pytest.approx(expected, rel=5e-13, abs=5e-324)


@pytest.mark.parametrize("x", [0.1, 0.5, 1 - 1e-12])
def test_uniform_and_half_shape_integrals(x):
    assert independent(3, 1.0, 1.0, x) == x
    assert independent(3, 1.0, 0.5, x) == pytest.approx(1 - (1 - x) ** 0.5, rel=2e-15)


@pytest.mark.parametrize("a", [0.1, 0.5, 1e-100, 5e-324])
def test_symmetric_beta_integral_midpoint(a):
    assert independent(3, a, a, 0.5) == pytest.approx(0.5, rel=2e-15)


def test_native_positive_tolerance_cases_satisfy_series_domains():
    with localcontext() as ctx:
        ctx.prec = 800
        for case in FIXTURE["cases"]:
            if case["eps"] <= 0:
                continue
            a, b, x, eps = (Decimal.from_float(case[k]) for k in ("a", "b", "x", "eps"))
            if case["mode"] == 1:
                assert a <= min(eps, eps * b) and b * x <= 1 and x <= Decimal(".5")
            elif case["mode"] == 2:
                assert b < min(eps, eps * a) and x <= Decimal(".5")
            else:
                assert b <= 1 or b * x <= Decimal.from_float(0.7)
