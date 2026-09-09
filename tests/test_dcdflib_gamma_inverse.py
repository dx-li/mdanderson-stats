"""Native inverse-gamma contracts and independent high-precision tail checks."""

import json
import math
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from test_cdflib_error_exponential_reference import independent_error
from test_cdflib_incomplete_gamma_reference import log_gamma

from mdanderson_stats import dcdflib_support as legacy

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_gamma_inverse.json").read_text())
CASES = [(lang, c) for lang, row in FIXTURE["profiles"].items() for c in row["cases"]]


@cache
def independent_log_tail(a, x, lower):
    """Decimal integral series; never round the tail to a float64 subnormal bin."""
    with localcontext() as ctx:
        ctx.prec = 800
        a, x = Decimal.from_float(a), Decimal.from_float(x)
        if a == a.to_integral_value():
            term = total = Decimal(1)
            for k in range(1, int(a)):
                term *= x / k
                total += term
            q = (-x).exp() * total
            return float((1 - q).ln() if lower else q.ln())
        log_r = a * x.ln() - x - log_gamma(a)
        if x >= 100:
            assert a <= 20
            term = total = Decimal(1)
            for k in range(1, 81):
                term *= (a - k) / x
                total += term
            assert abs(term * (a - 81) / x) < abs(total) * Decimal("1e-35")
            logq = log_r + total.ln() - x.ln()
            return float((1 - logq.exp()).ln() if lower else logq)
        term = total = Decimal(1)
        for k in range(1, 10000):
            term *= x / (a + k)
            total += term
            if term < total * Decimal("1e-750"):
                break
        else:
            raise AssertionError("Independent gamma integral did not converge")
        logp = log_r + total.ln() - a.ln()
        return float(logp if lower else (1 - logp.exp()).ln())


@pytest.mark.parametrize("language,case", CASES)
def test_native_contract_and_independent_gamma_integral(language, case):
    a, p, q, x0 = (case[k] for k in ["a", "p", "q", "x0"])
    if a <= 0 or p + q != 1:
        with pytest.raises(ValueError):
            legacy.gaminv(a, p, q, x0)
        return
    if p == 0 or q == 0:
        assert legacy.gaminv(a, p, q, x0) == case["result"]
        assert case["status"] == 0
        return
    underflows = (a <= 1e-13 and (q >= 0.5 or q / a > 1000)) or (a == 0.5 and p < 1e-200)
    if underflows:
        with pytest.raises(ValueError, match="not representable"):
            legacy.gaminv(a, p, q, x0)
        return
    actual = float(legacy.gaminv(a, p, q, x0))
    assert actual > 0 and math.isfinite(actual)
    if a >= 1e20:
        if p == 0.5 or a >= 1e100:
            if a >= 1e100:
                # Every positive float64 tail has |normal deviate|<39.
                assert 39 * math.sqrt(a) + 508 < np.spacing(a) / 2
            assert actual == a
        else:
            # Gamma standardized quantiles approach normal quantiles with
            # O(z²/sqrt(a)) correction. Float64 x spacing here contributes <2e-6.
            z = (actual - a) / math.sqrt(a)
            normal_log_tail = (
                math.log(independent_error(abs(z) / math.sqrt(2), scaled=True))
                - z * z / 2
                - math.log(2)
            )
            # x rounding and the O(z²/sqrt(a)) gamma correction together
            # contribute less than 1e-4 in the log tail at these a=1e20 cases.
            assert abs(normal_log_tail - math.log(min(p, q))) < 1e-4
            assert (z < 0) == (p < q)
        return
    log_actual = independent_log_tail(a, actual, p <= q)
    assert abs(log_actual - math.log(min(p, q))) < 1e-8
    if case["kind"] == "ordinary" and case["status"] >= 0:
        np.testing.assert_allclose(actual, case["result"], rtol=1e-9, atol=0)


def test_exact_and_inaccurate_positive_hints():
    # q=exp(-1), a=1 has exact x=1. Also retain a nearby admissible approximation.
    q = math.exp(-1)
    near = 1 + 1e-13
    assert legacy.gaminv(1, None, q, 1) == 1
    assert legacy.gaminv(1, None, q, near) == near
    for hint in [-1, 0, 1e-300, 100, 1e300]:
        assert float(legacy.gaminv(1, None, q, hint)) == pytest.approx(1, rel=1e-14)


@pytest.mark.parametrize("a", [1e-12, 1e-6, 0.001, 0.1, 0.25, 0.5, 0.75, 0.999999])
@pytest.mark.parametrize("q", [5e-324, 1e-320, 1e-310, np.finfo(float).tiny])
def test_subnormal_upper_tail_coordinates_against_log_integral(a, q):
    x = float(legacy.gaminv(a, None, q))
    assert abs(independent_log_tail(a, x, False) - math.log(q)) < 3e-11


def test_broadcasts_endpoints_ownership_and_tail_completion():
    a = np.array([[1.0], [2.0]])
    p = np.array([0.0, 0.5, 1.0])
    out = legacy.gaminv(a, p, x0=np.array([1, 2, 3]))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out[:, 0], 0)
    np.testing.assert_array_equal(out[:, 2], np.finfo(float).max)
    np.testing.assert_allclose(legacy.gaminv(a, None, 1 - p), out, rtol=1e-14)
    assert not out.flags.writeable and not np.shares_memory(out, a)
    with pytest.raises(ValueError):
        out.setflags(write=True)
    assert legacy.gaminv(np.empty((0, 1)), np.ones((1, 3))).shape == (0, 3)
    with np.errstate(all="raise"):
        assert legacy.gaminv(1, 5e-324, 1) == 5e-324
        assert legacy.gaminv(1e308, 0.5, x0=1e308) == 1e308


@pytest.mark.parametrize("a", [0, -1, np.inf, -np.inf, np.nan])
def test_invalid_shape(a):
    with pytest.raises(ValueError):
        legacy.gaminv(a, 0.5)


@pytest.mark.parametrize("x0", [np.inf, -np.inf, np.nan])
def test_invalid_hint_even_at_endpoints(x0):
    with pytest.raises(ValueError):
        legacy.gaminv(1, 0, x0=x0)


@pytest.mark.parametrize("p,q", [(None, None), (-1, 2), (0.1, 0.1), (np.nan, 0.5), (1, np.inf)])
def test_invalid_probabilities(p, q):
    with pytest.raises(ValueError):
        legacy.gaminv(1, p, q)
