"""Independent expectations for the remaining native beta helpers."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from test_cdflib_beta_factors_reference import log_beta
from test_cdflib_beta_series_reference import independent

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_beta_remaining.json").read_text())


def integral(a, b, x):
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    if a == 0:
        return 1.0
    if b == 0:
        return 0.0
    if a == b and x == Decimal(".5"):
        return 0.5
    if a == b and a >= 2:
        delta = Decimal(".5") - x
        u = 4 * delta * delta
        if a * u <= 2 or (a >= Decimal("1e100") and a * u <= 1000):
            # Integrate the symmetric density about its midpoint. The
            # coordinate displacement remains exact in Decimal arithmetic.
            # For enormous shapes, a*u<=1000 bounds absolute coefficient
            # growth by exp(1000), leaving enough precision even for subnormals.
            term = total = Decimal(1)
            for n in range(1, 10000):
                term *= (n - a) * u / n
                add = term / (2 * n + 1)
                total += add
                if abs(add) < abs(total) * Decimal("1e-750"):
                    break
            else:
                raise AssertionError("symmetric central integral did not converge")
            log_scale = abs(delta).ln() + (2 - 2 * a) * Decimal(2).ln() - log_beta(a, a)
            decrement = log_scale.exp() * total
            return float(Decimal(".5") - decrement if delta > 0 else Decimal(".5") + decrement)
    if x > Decimal(".5"):
        # Ask directly for the upper tail of the reflected small-coordinate integral.
        if a * (1 - x) <= 100 or a <= 1:
            return independent(1, b, a, 1 - x)
    else:
        if b * x <= 100 or b <= 1:
            return independent(2, a, b, x)
    # Integrand monotonicity bounds the omitted (1-t)**(b-1) factor.
    log_bound = a * x.ln() - a.ln() - log_beta(a, b)
    if b < 1:
        log_bound += (b - 1) * (1 - x).ln()
    if log_bound < -800:
        return 0.0
    if x > Decimal(".5"):
        return 1 - integral(b, a, 1 - x)
    raise AssertionError("case needs another independent integration method")


def expected(case):
    with localcontext() as ctx:
        ctx.prec = 800
        a, b, x, y, lam, initial = (
            Decimal.from_float(case[k]) for k in ("a", "b", "x", "y", "lam", "initial")
        )
        if case["mode"] == 1:
            x = (a - lam) / (a + b)
        else:
            x = x if x <= y else 1 - y
        return float(initial) + integral(a, b, x) if case["mode"] == 3 else integral(a, b, x)


ERRORS = {
    (-1.0, 1.0, 0.5, 0.5): 1,
    (0.0, 0.0, 0.5, 0.5): 2,
    (1.0, 1.0, -0.1, 1.1): 3,
    (1.0, 1.0, 0.5, -0.1): 4,
    (1.0, 1.0, 0.5, 0.4): 5,
    (0.0, 1.0, 0.0, 1.0): 6,
    (1.0, 0.0, 1.0, 0.0): 7,
}


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_against_independent_integrals_and_contract(case):
    mode, a, b, x, y, lam = (case[k] for k in ("mode", "a", "b", "x", "y", "lam"))
    if mode == 4 and (a, b, x, y) in ERRORS:
        assert case["execution_outcome"] == "completed"
        assert case["status"] == ERRORS[a, b, x, y]
        assert case["result"] == [0.0, 0.0]
        return
    target = expected(case)
    if case["execution_outcome"] == "timeout":
        assert (mode == 2 and ((a, b, x) == (20.0, 50.0, 0.9) or a == b == 1e308)) or (
            mode == 4 and a == b == 1e308 and 0 < x < 1
        )
        assert 0 <= target <= 1
        return
    assert case["execution_outcome"] == "completed"
    value = case["result"][0]
    if mode == 3 and case["status"] == 1:
        assert x == 1 or (a == 1e308 and x >= 0.5)
        assert value == case["initial"]
        assert target == case["initial"] + (1 if x == 1 else 0)
    elif (mode == 1 and lam == a) or (mode == 3 and (x == 0 or (a == 1e308 and x == 0.1))):
        assert case["status"] == 0 and value == "nan" and target == 0
    elif mode == 1 and a == b == 1e308 and lam in [1e140, 1e146]:
        assert value == 0.5 and target < 0.5
        assert 0.55 * lam / 1e154 < value - target < 0.58 * lam / 1e154
    elif mode == 1 and (a, b, lam) == (50.0, 15.0, 25.0):
        assert 1e-8 < abs(value / target - 1) < 1.1e-8
    elif mode == 2 and x == 1:
        assert value == 0 and target == 1
    elif mode == 2 and (a, b, x) == (15.0, 15.0, 0.9):
        assert value > 1 and target < 1
    elif mode == 2 and (a, b, x) == (20.0, 50.0, 0.5):
        assert 1e-11 < abs(value - target) < 1e-10
    elif mode == 2 and (a, b, x) == (100.0, 100.0, 0.9):
        assert 0 < value < 1e-40 and target == 1
    else:
        assert case["status"] == 0
        assert value == pytest.approx(target, rel=5e-13, abs=5e-324)
    if mode == 4:
        reflected = dict(case, a=b, b=a, x=y, y=x)
        assert case["result"][1] == pytest.approx(expected(reflected), rel=5e-13, abs=5e-324)


@pytest.mark.parametrize("a,b", [(2, 3), (15, 50), (50, 15), (100, 100)])
@pytest.mark.parametrize("x", [0.01, 0.1, 0.5, 0.9])
def test_extended_integral_against_exact_integer_binomial_sum(a, b, x):
    with localcontext() as ctx:
        ctx.prec = 800
        xx = Decimal.from_float(x)
        n = a + b - 1
        exact = sum(Decimal(math.comb(n, k)) * xx**k * (1 - xx) ** (n - k) for k in range(a, n + 1))
        actual = integral(Decimal(a), Decimal(b), xx)
        assert actual == pytest.approx(float(exact), rel=2e-15, abs=5e-324)


def test_source_domains_and_recorded_status_coverage():
    statuses = set()
    for c in FIXTURE["cases"]:
        if c["mode"] == 1:
            assert c["a"] >= 15 and c["b"] >= 15 and 0 <= c["lam"] <= c["a"]
        elif c["mode"] == 2:
            assert c["a"] > 1 and c["b"] > 1 and 0 <= c["x"] <= 1
            # The source states this relation without requiring nonnegative lambda.
            assert c["lam"] == pytest.approx(
                c["a"] * c["y"] - c["b"] * c["x"], rel=1e-15, abs=1e-15
            )
        elif c["mode"] == 3:
            assert c["a"] >= 15 and 0 < c["b"] <= 1 and 0 <= c["x"] <= 1
        elif c["execution_outcome"] == "completed":
            statuses.add(c["status"])
    assert statuses == set(range(8))


@pytest.mark.parametrize("a,x", [(15, 0.4), (100, 0.45), (1000, 0.49)])
def test_central_integral_against_positive_binomial_sum(a, x):
    with localcontext() as ctx:
        ctx.prec = 800
        xx = Decimal.from_float(x)
        n = 2 * a - 1
        exact = sum(Decimal(math.comb(n, k)) * xx**k * (1 - xx) ** (n - k) for k in range(a, n + 1))
        assert integral(Decimal(a), Decimal(a), xx) == pytest.approx(float(exact), rel=2e-15)
