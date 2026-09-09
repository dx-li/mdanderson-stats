"""High-precision log-gamma identities for public beta support procedures."""

import json
import math
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path

import pytest
from test_cdflib_error_exponential_reference import decimal_pi
from test_cdflib_gamma_support_reference import EULER, positive_log_gamma, positive_psi

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_beta_support.json").read_text())


@cache
def oracle_log_gamma(x):
    # Caller uses 800 digits. Keep tiny offsets from both exact roots; a
    # Stirling approximation's absolute error would conceal these offsets.
    for root in [1, 2]:
        d = x - root
        if abs(d) < Decimal("1e-20"):
            result = -EULER * d + decimal_pi() ** 2 / 12 * d * d
            if root == 2:
                result += (1 + d).ln()
            return result
    return positive_log_gamma(x)


@cache
def independent(mode, a, b):
    with localcontext() as ctx:
        ctx.prec = 800
        a, b = Decimal.from_float(a), Decimal.from_float(b)
        if mode == 1:
            if abs(a) / b < Decimal("1e-30"):
                # The next relative term is O(abs(a)/b). A derivative avoids
                # subtracting two Stirling errors with different recurrence shifts.
                return float(-a * positive_psi(b))
            return float(oracle_log_gamma(b) - oracle_log_gamma(a + b))
        if mode == 2:

            def delta(x):
                return (
                    oracle_log_gamma(x)
                    - (x - Decimal(".5")) * x.ln()
                    + x
                    - (2 * decimal_pi()).ln() / 2
                )

            return float(delta(a) + delta(b) - delta(a + b))
        if mode in [3, 4]:
            return float(oracle_log_gamma(a) + oracle_log_gamma(b) - oracle_log_gamma(a + b))
        if mode == 5:
            return float(oracle_log_gamma(a + b))
        return float(
            oracle_log_gamma(b + 1) - oracle_log_gamma(a + 1) - oracle_log_gamma(b - a + 1)
        )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_against_independent_gamma_identities(case):
    assert case["execution_outcome"] == "completed"
    mode, a, b, result = case["mode"], case["a"], case["b"], case["result"]
    if mode == 1 and a + b <= 0:
        assert result == "nan"
        return
    if mode in [3, 4] and min(a, b) <= 0:
        assert result in [-1, 0]  # Domain sentinels propagate as ordinary finite answers.
        return
    if mode == 6 and (a <= -1 or b - a <= -1 or b <= -1):
        assert result == "inf" if b == -1 else math.isfinite(result)
        return
    expected = independent(mode, a, b)
    if math.isinf(expected):
        assert result == str(expected)
    elif mode == 1 and a > 0 and a / b == 0:
        assert result == pytest.approx(-a * (math.log(b) - 1), rel=1e-14, abs=5e-324)
        if a >= 1e-100:
            assert abs(result - expected) > abs(expected) * 1e-4
    elif mode == 1 and a == 1e-10 and b == 1e308:
        assert abs(result - expected) > abs(expected) * 1e-10  # Rounded subnormal a/b.
        assert result == pytest.approx(-b * (a / b) - a * (math.log(b) - 1), rel=1e-14)
    elif mode == 1 and a in (-8.0, -9.5):
        assert b == 10 and a + b > 0
        assert abs(result - expected) > (1e-7 if a == -8 else 1)
    elif mode in (3, 4) and {a, b} == {1.0, 1 + 2**-52}:
        assert result == pytest.approx(-float(EULER) * 2**-52, rel=1e-14)
        assert abs(result - expected) > abs(expected) * 0.4
    elif mode == 5 and {a, b} == {1.0, 1 + 2**-52}:
        assert result == 0 and expected > 0
    elif mode == 6 and b == 1 and 0 < a < 1e-90:
        assert result == 0 and expected > 0
    else:
        assert result == pytest.approx(expected, rel=4e-13, abs=5e-324)


@pytest.mark.parametrize("n,k", [(10, 0), (10, 1), (10, 5), (10, 10), (50, 25)])
def test_oracle_against_exact_binomial_coefficient(n, k):
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float(Decimal(math.comb(n, k)).ln())
    assert independent(6, float(k), float(n)) == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0, 8.0, 1e100, 1e308])
def test_oracle_beta_unit_shape_identity(b):
    with localcontext() as ctx:
        ctx.prec = 800
        expected = -float(Decimal.from_float(b).ln())
    assert independent(3, 1.0, b) == pytest.approx(expected, rel=2e-15, abs=0)


def test_oracle_half_shape_and_gamma_ratio_identities():
    assert independent(3, 0.5, 0.5) == pytest.approx(math.log(math.pi), rel=2e-15)
    assert independent(1, 1.0, 8.0) == pytest.approx(-math.log(8), rel=2e-15)
    assert independent(1, -1.0, 10.0) == pytest.approx(math.log(9), rel=2e-15)
    assert independent(6, 1e-100, 1.0) == pytest.approx(1e-100, rel=2e-15, abs=0)
    assert independent(6, 5e-324, 1.0) == 5e-324


def test_native_renamed_beta_entry_points_agree():
    results = {
        mode: {(c["a"], c["b"]): c["result"] for c in FIXTURE["cases"] if c["mode"] == mode}
        for mode in (3, 4)
    }
    assert results[3] == results[4]
