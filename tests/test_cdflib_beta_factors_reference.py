"""Independent beta definitions distinguish native factor/recurrence failures."""

import json
import math
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path

import pytest
from test_cdflib_beta_support_reference import oracle_log_gamma

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_beta_factors.json").read_text())


def log_gamma(a):
    return oracle_log_gamma(1 + a) - a.ln() if a < Decimal("1e-20") else oracle_log_gamma(a)


def log_beta(a, b):
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@cache
def beta_cdf(a, b, x, y):
    if a == a.to_integral_value() and b == b.to_integral_value():
        total = int(a + b - 1)
        assert total < 10000
        return sum(
            Decimal(math.comb(total, k)) * x**k * y ** (total - k) for k in range(int(a), total + 1)
        )
    assert b <= 1 and x <= Decimal(".5")
    term = total = Decimal(1)
    for k in range(1, 10000):
        term *= (k - b) * x / k
        add = term * a / (a + k)
        total += add
        if abs(add) < abs(total) * Decimal("1e-750"):
            break
    else:
        raise AssertionError("independent beta integral series did not converge")
    return (a * x.ln() - a.ln() - log_beta(a, b)).exp() * total


@cache
def independent(mode, a, b, x, y, mu=0, n=1):
    with localcontext() as ctx:
        ctx.prec = 800
        a, b = Decimal.from_float(a), Decimal.from_float(b)
        # Coordinates encode complementary probabilities. Preserve the smaller
        # one, including tails whose large complement rounds to exactly one.
        if x <= y:
            x = Decimal.from_float(x)
            y = 1 - x
        else:
            y = Decimal.from_float(y)
            x = 1 - y
        if x == 0 or y == 0:
            return 0.0
        log_factor = a * x.ln() + b * y.ln() - log_beta(a, b)
        if mode < 3:
            exponent = log_factor + (mu if mode == 2 else 0)
            if exponent > 1000:
                return math.inf
            return 0.0 if exponent < -1000 else float(exponent.exp())
        if b == 1:
            return float((a * x.ln()).exp() * (1 - (n * x.ln()).exp()))
        if a >= Decimal("1e100"):
            # For equal huge shapes, fixed n<=100 changes each recurrence
            # summand by relative O(n**2/a), below float64 resolution.
            assert a == b and x == y and n <= 100
            return float((log_factor - a.ln()).exp() * n)
        return float(beta_cdf(a, b, x, y) - beta_cdf(a + n, b, x, y))


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_beta_factors_against_independent_values(case):
    a, b, x, y, mode = (case[k] for k in ("a", "b", "x", "y", "mode"))
    if case["execution_outcome"] == "timeout":
        assert mode == 3 and case["n"] == 2**31 - 1
        assert case["timeout_seconds"] == 3 and a == b == 1
        assert independent(mode, a, b, x, y, n=case["n"]) > 0
        return
    assert case["execution_outcome"] == "completed"
    if a <= 0:
        assert mode == 1 and b == 1 and x == y == 0.5
        expected = 0 if a == 0 else a * x**a * y
        assert case["result"] == pytest.approx(expected, rel=5e-13)
        return
    if abs(x + y - 1) > 1e-15:
        assert mode == 1
        if a == 2:
            expected = independent(1, a, b, x, 1 - x)
        else:
            assert a == b == 8
            expected = x**8 * y**8 * math.factorial(15) / math.factorial(7) ** 2
        assert case["result"] == pytest.approx(expected, rel=5e-13)
        return
    if mode == 3 and case["n"] <= 0:
        assert case["result"] == pytest.approx(independent(1, a, b, x, y), rel=5e-13)
        return
    expected = independent(mode, a, b, x, y, case["mu"], case["n"])
    if a == b == 1e308:
        assert case["result"] == "nan" and math.isfinite(expected) and expected > 0
        return
    if mode == 2 and case["mu"] == 1000 and ((a == 1 and x == 1e-300) or a == 1e-309):
        assert case["result"] == "inf" and math.isfinite(expected) and expected > 0
        return
    if mode == 2 and case["mu"] == -744 and a >= 1e100:
        rounded_first = independent(1, a, b, x, y) * math.exp(-744)
        assert case["result"] == pytest.approx(rounded_first, rel=5e-13)
        assert 0.28 < case["result"] / expected - 1 < 0.29
        return
    if mode == 2 and case["mu"] == -1000 and a == 1e300:
        assert case["result"] == 0 and expected > 0
        return
    if mode == 3 and case["eps"] == 1e-3:
        assert 0.009 < 1 - case["result"] / expected < 0.010
        return
    if math.isinf(expected):
        assert case["result"] == "inf"
    else:
        assert case["result"] == pytest.approx(expected, rel=5e-13, abs=5e-324)


@pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
def test_unit_beta_factor_and_shift_identities(x):
    y = 1 - x
    assert independent(1, 1.0, 1.0, x, y) == pytest.approx(x * y, rel=2e-15)
    assert independent(3, 1.0, 1.0, x, y, n=10) == pytest.approx(x * (1 - x**10), rel=2e-15)


@pytest.mark.parametrize("x", [0.1, 0.5])
def test_half_shape_difference_against_integral_identity(x):
    expected = 2 * math.sqrt(x * (1 - x)) / math.pi
    assert independent(3, 0.5, 0.5, x, 1 - x) == pytest.approx(expected, rel=3e-15)


def test_large_unit_shift_has_a_finite_closed_form():
    value = independent(3, 1.0, 1.0, 1 - 1e-12, 1e-12, n=2**31 - 1)
    expected = (1 - 1e-12) * -math.expm1((2**31 - 1) * math.log1p(-1e-12))
    assert value == pytest.approx(expected, rel=3e-15)
