"""Independent defining functions expose native cutoffs and intermediate overflow."""

import json
import math
import sys
from decimal import Decimal, localcontext
from functools import cache, lru_cache
from pathlib import Path

import pytest

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_error_exponential.json").read_text())


@lru_cache(maxsize=1)
def decimal_pi():
    # Gauss–Legendre iteration, independent of the source rational approximations.
    with localcontext() as ctx:
        ctx.prec = 1100
        a, b, t, p = Decimal(1), 1 / Decimal(2).sqrt(), Decimal("0.25"), Decimal(1)
        for _ in range(13):
            mean = (a + b) / 2
            b = (a * b).sqrt()
            t -= p * (a - mean) ** 2
            a = mean
            p *= 2
        return (a + b) ** 2 / (4 * t)


@cache
def independent_error(x, scaled=False, complement=False):
    with localcontext() as ctx:
        ctx.prec = 1100
        z = Decimal.from_float(abs(x))
        if z >= 30:
            # Alternating erfcx asymptotic expansion; 40 terms at x>=30
            # bound the omitted relative term below 1e-70.
            term = total = Decimal(1)
            for n in range(1, 40):
                term *= -Decimal(2 * n - 1) / (2 * z * z)
                total += term
            positive_scaled = total / (decimal_pi().sqrt() * z)
            positive_complement = Decimal(0)  # erfc(30) is already below float range.
        else:
            # erf(x)=2/sqrt(pi) sum (-1)^n*x^(2n+1)/(n!*(2n+1)).
            # Extra precision absorbs cancellation at x near 28.
            term = total = z
            for n in range(1, 10000):
                term *= -z * z / n
                add = term / (2 * n + 1)
                total += add
                if abs(add) < Decimal("1e-1000"):
                    break
            else:
                raise AssertionError("independent series did not converge")
            positive_complement = 1 - 2 * total / decimal_pi().sqrt()
            positive_scaled = (z * z).exp() * positive_complement
        if scaled:
            if x < 0:
                if z >= 27:
                    return math.inf
                return float(2 * (z * z).exp() - positive_scaled)
            return float(positive_scaled)
        if complement:
            return float(2 - positive_complement if x < 0 else positive_complement)
        return float((1 - positive_complement) * (-1 if x < 0 else 1))


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] <= 2])
def test_native_error_functions_against_independent_series(case):
    assert case["execution_outcome"] == "completed"
    mode, ind, x = case["mode"], case["ind"], case["x"]
    expected = independent_error(x, scaled=mode == 2 and ind != 0, complement=mode == 2)
    if math.isinf(expected):
        assert case["result"] == "inf"
    elif mode == 2 and ind == 0 and 26.7 <= x <= 27.2:
        assert case["result"] == 0
        assert expected > 0  # Source discards representable subnormal tails.
    else:
        assert case["result"] == pytest.approx(expected, rel=2e-13, abs=5e-324)


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 3])
def test_native_exponential_sum_against_decimal(case):
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float((Decimal(case["ind"]) + Decimal.from_float(case["x"])).exp())
    pair = (case["ind"], case["x"])
    if pair in [(1000, -999.5), (-1000, 999.5)]:
        assert case["result"] == "nan" and math.isfinite(expected)
    elif pair == (710, -1.0):
        assert case["result"] == "inf" and math.isfinite(expected)
    elif pair == (-745, 1.0):
        assert case["result"] == 1.5e-323
        assert expected == 1e-323  # Product rounds twice after exp(-745) underflows.
    elif math.isinf(expected):
        assert case["result"] == "inf"
    else:
        assert case["result"] == pytest.approx(expected, rel=3e-15, abs=0)


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 4])
def test_exparg_is_normal_range_not_nonzero_range(case):
    expected = math.log(sys.float_info.max if case["ind"] == 0 else sys.float_info.min)
    assert case["result"] == expected
    if case["ind"]:
        assert math.exp(case["result"] - 1) > 0
        assert math.log(math.ulp(0.0)) < case["result"]
