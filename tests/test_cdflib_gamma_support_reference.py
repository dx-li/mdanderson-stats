"""Independent recurrences and Stirling series distinguish native failures."""

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from pathlib import Path

import pytest
from test_cdflib_error_exponential_reference import decimal_pi

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_gamma_support.json").read_text())
EULER = Decimal(
    "0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709"
)


@cache
def bernoulli_even():
    values = [Fraction(1)]
    for n in range(1, 67):
        values.append(-sum(math.comb(n + 1, k) * values[k] for k in range(n)) / (n + 1))
    return tuple(Decimal(v.numerator) / Decimal(v.denominator) for v in values[2::2])


def positive_log_gamma(x):
    if x in (1, 2):
        return Decimal(0)
    correction = Decimal(0)
    while x < 50:
        correction -= x.ln()
        x += 1
    result = (x - Decimal(".5")) * x.ln() - x + (2 * decimal_pi()).ln() / 2
    for k, b in enumerate(bernoulli_even()[:32], 1):
        result += b / (2 * k * (2 * k - 1) * x ** (2 * k - 1))
    return result + correction


def positive_psi(x):
    correction = Decimal(0)
    while x < 50:
        correction -= 1 / x
        x += 1
    result = x.ln() - 1 / (2 * x)
    for k, b in enumerate(bernoulli_even()[:32], 1):
        result -= b / (2 * k * x ** (2 * k))
    return result + correction


@cache
def independent(mode, x):
    with localcontext() as ctx:
        ctx.prec = 100
        z = Decimal.from_float(x)
        if mode in (4, 5):
            if z in (0, 1):
                return 0.0
            # The first neglected term is O(z²). This branch also preserves
            # arguments too small to survive forming 1+z at oracle precision.
            if abs(z) < Decimal("1e-30"):
                return float((-EULER if mode == 4 else EULER) * z)
            log_value = positive_log_gamma(1 + z)
            return float(log_value if mode == 4 else (-log_value).exp() - 1)
        if mode == 7:
            if z < 0 and z % 1 == Decimal("-.5"):
                return float(positive_psi(1 - z))  # cot(pi*x)=0 at half integers.
            if z < 0:
                return float(positive_psi(1 + z) - 1 / z)
            return float(positive_psi(z))
        if z > 0:
            log_value = positive_log_gamma(z)
            if mode != 6:
                return float(log_value)
            return math.inf if log_value > 1000 else float(log_value.exp())
        # Recorded negative nonintegers are half-integers or near zero.
        if z % 1 == Decimal("-.5"):
            log_value = decimal_pi().ln() - positive_log_gamma(1 - z)
            sign = -1 if int(-z) % 2 == 0 else 1
            if mode == 1:
                assert sign > 0
                return float(log_value)
            if log_value < -1000:
                return math.copysign(0.0, sign)
            return sign * float(log_value.exp())
        return float(positive_log_gamma(1 + z).exp() / z)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_gamma_support_against_independent_definitions(case):
    mode, x = case["mode"], case["x"]
    if case["execution_outcome"] == "timeout":
        assert mode == 1 and x == -1e100 and case["timeout_seconds"] == 3
        return
    assert case["execution_outcome"] == "completed"
    result = case["result"]
    if x <= 0 and mode in (1, 2, 3, 6, 7):
        if mode == 2:
            assert result == -1
            return
        if mode == 3:
            assert result == ("inf" if x == 0 else "nan")
            return
        if x == math.floor(x):
            if mode in (6, 7):
                assert result == 0  # Pole sentinel, not a mathematical value.
            else:
                assert result in ("inf", "nan")
            return
        if mode == 1 and int(-x) % 2 == 0:
            assert result in ("nan", "-inf")  # No real logarithm of negative gamma.
            return
    expected = independent(mode, x)
    if mode == 1 and 0 < x < 1e-308:
        assert result == "inf" and math.isfinite(expected)
    elif mode == 1 and x == -175.5:
        assert 1e-6 < abs(result - expected) < 1e-5  # Logging a rounded subnormal.
    elif mode == 1 and x == -177.5:
        assert result == "-inf" and math.isfinite(expected)  # Product underflows first.
    elif mode == 6 and math.isinf(expected):
        assert result == 0  # Native zero sentinel hides overflow.
    elif math.isinf(expected):
        assert result == str(expected)
    elif mode == 6 and x in (-172.5, -175.5, -177.5):
        assert result == 0 and expected != 0  # Positive gamma intermediate overflows.
    elif mode == 7 and x == -2147483647.5:
        assert result == 0 and expected > 20  # Integer-machine cutoff rejects valid input.
    elif mode == 7 and x == 1e10:
        assert result == math.log(x)
        assert result - expected == pytest.approx(1 / (2 * x), rel=1e-4)
    elif mode == 1 and (abs(x - 1) < 1e-15 or x == 2):
        assert result > 0  # Rational approximation loses the root and its sign.
        assert abs(result - expected) < 5e-15
    else:
        assert result == pytest.approx(expected, rel=4e-13, abs=5e-324)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 50, 170])
def test_independent_oracle_matches_exact_factorials(n):
    with localcontext() as ctx:
        ctx.prec = 100
        value = Decimal(math.factorial(n - 1))
        assert independent(2, float(n)) == pytest.approx(float(value.ln()), rel=2e-15, abs=0)
        assert independent(6, float(n)) == pytest.approx(float(value), rel=2e-15, abs=0)
        harmonic = sum((Decimal(1) / k for k in range(1, n)), Decimal(0))
        assert independent(7, float(n)) == pytest.approx(float(harmonic - EULER), rel=2e-15)


@pytest.mark.parametrize("x,multiplier", [(-0.5, -2), (-1.5, 4 / 3), (0.5, 1), (1.5, 0.5)])
def test_independent_oracle_matches_half_integer_gamma(x, multiplier):
    assert independent(6, x) == pytest.approx(multiplier * math.sqrt(math.pi), rel=2e-15)


def test_stirling_omitted_terms_are_below_required_absolute_accuracy():
    with localcontext() as ctx:
        ctx.prec = 100
        b66 = abs(bernoulli_even()[32])
        assert b66 / (66 * 65 * Decimal(50) ** 65) < Decimal("1e-65")
        assert b66 / (66 * Decimal(50) ** 66) < Decimal("1e-65")
