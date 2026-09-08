"""Native, rational-probability and analytic beta inversion validation."""

import itertools
import json
from fractions import Fraction
from math import comb, expm1, log, log1p
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_beta, cdf_beta, cum_beta, inv_beta

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_beta.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_beta(case):
    which, p, q, x, cx, a, b = case["input"]
    kwargs = dict(cum=p, ccum=q, x=x, cx=cx, a=a, b=b)
    for name in {1: ("cum", "ccum"), 2: ("x", "cx"), 3: ("a",), 4: ("b",)}[which]:
        kwargs.pop(name)
    result = cdf_beta(which, **kwargs)
    if which == 1:
        assert case["status"] == 0
        np.testing.assert_allclose(
            [result.cum, result.ccum], case["result"][:2], rtol=3e-11, atol=1e-15
        )
    elif which == 2:
        # The generating x is known. Native inversion can return a stale pair,
        # even with success status; do not promote that output to ground truth.
        np.testing.assert_allclose([result.x, result.cx], [x, cx], rtol=2e-10, atol=1e-14)
    else:
        actual, expected = (result.a, a) if which == 3 else (result.b, b)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-14)


@pytest.mark.parametrize(
    "a,b,numerator", list(itertools.product([1, 2, 5, 12], [1, 3, 7], [1, 3, 7, 9]))
)
def test_integer_beta_rational_binomial_identity(a, b, numerator):
    x = Fraction(numerator, 10)
    n = a + b - 1
    expected = sum(Fraction(comb(n, k)) * x**k * (1 - x) ** (n - k) for k in range(a, n + 1))
    result = cdf_beta(x=float(x), a=a, b=b)
    assert float(result.cum) == pytest.approx(float(expected), rel=1e-12, abs=1e-16)
    assert float(result.ccum) == pytest.approx(float(1 - expected), rel=1e-12, abs=1e-16)


@pytest.mark.parametrize("shape", [1e-10, 0.2, 1.0, 5.0, 100.0, 1e10])
def test_one_shape_analytic_quantiles(shape):
    p = np.array([0.0, 1e-12, 0.2, 0.8, 1.0])
    result = cdf_beta(2, cum=p, a=shape, b=1)
    np.testing.assert_allclose(result.x, p ** (1 / shape), rtol=1e-12, atol=1e-15)
    reflected = cdf_beta(2, ccum=p, a=1, b=shape)
    np.testing.assert_allclose(reflected.cx, p ** (1 / shape), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "x,a", list(itertools.product([0.01, 0.2, 0.8, 0.99], [0.1, 1.0, 5.0, 20.0]))
)
def test_analytic_shape_inversion(x, a):
    p = x**a
    q = -expm1(a * log(x))
    result = cdf_beta(3, x=x, b=1, cum=p, ccum=q)
    assert float(result.a) == pytest.approx(a, rel=2e-11)
    result = cdf_beta(4, cx=x, a=1, cum=q, ccum=p)
    assert float(result.b) == pytest.approx(a, rel=2e-11)


def test_extreme_coordinate_and_probability_complements():
    for tiny in [1e-20, 1e-100, 1e-300]:
        direct = cdf_beta(x=1, cx=tiny, a=1, b=1)
        assert float(direct.ccum) == pytest.approx(tiny, rel=1e-12, abs=0)
        result = cdf_beta(2, cum=1, ccum=tiny, a=1, b=1)
        assert float(result.cx) == pytest.approx(tiny, rel=1e-12, abs=0)
        assert float(result.x) == 1
    q = -expm1(1e5 * log1p(-1e-20))
    found = cdf_beta(3, x=1, cx=1e-20, b=1, ccum=q)
    assert float(found.a) == pytest.approx(1e5, rel=1e-10)
    p = -expm1(2e-10 * log(1e-100))
    found = cdf_beta(4, x=1, cx=1e-100, a=1, cum=p)
    assert float(found.b) == pytest.approx(2e-10, rel=1e-8)


def test_broadcast_convenience_and_immutable_ownership():
    x = np.array([[0.1], [0.7]])
    a = np.array([0.2, 1, 5])
    r = cdf_beta(x=x, a=a, b=2)
    assert r.x.shape == (2, 3)
    np.testing.assert_array_equal(cum_beta(x, a, 2), r.cum)
    np.testing.assert_array_equal(ccum_beta(x, a, 2), r.ccum)
    np.testing.assert_allclose(inv_beta(r.cum, a, 2, ccum=r.ccum), r.x, rtol=1e-12)
    np.testing.assert_allclose(ccum_beta(None, a, 2, cx=1 - x), r.ccum, rtol=1e-12)
    x[:] = 0
    assert np.all(r.x > 0)
    for name in ("cum", "ccum", "x", "cx", "a", "b"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    inverted = cdf_beta(3, x=r.x, cx=r.cx, cum=r.cum, ccum=r.ccum, b=2)
    np.testing.assert_allclose(inverted.a, r.a, rtol=1e-10)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (0, {}),
        (True, {}),
        (1, {"x": 0.5, "a": 0, "b": 1}),
        (1, {"x": 0.5, "a": 1e11, "b": 1}),
        (1, {"x": 0.5, "a": float("nan"), "b": 1}),
        (1, {"a": 1, "b": 1}),
        (1, {"x": 0.5, "b": 1}),
        (1, {"x": 0.5, "cx": 0.4, "a": 1, "b": 1}),
        (1, {"x": 1.1, "a": 1, "b": 1}),
        (1, {"x": 0.5, "a": 1, "b": 1, "cum": 0.5}),
        (2, {"cum": 0.2, "ccum": 0.2, "a": 1, "b": 1}),
        (2, {"cum": 0.5, "a": 1, "b": 1, "x": 0.5}),
        (3, {"cum": 0, "x": 0.2, "b": 1}),
        (3, {"cum": 0.5, "x": 0, "b": 1}),
        (4, {"cum": 1, "x": 0.2, "a": 1}),
        (3, {"cum": 0.99, "x": 1e-300, "b": 1e-10}),
    ],
)
def test_invalid_or_unidentifiable_requests(which, kwargs):
    with pytest.raises(ValueError):
        cdf_beta(which, **kwargs)


def test_empty_broadcast_and_search_bounds():
    result = cdf_beta(3, cum=[], x=[], b=1)
    assert result.a.shape == (0,)
    for a in (1e-10, 1e10):
        original = cdf_beta(x=0.5, a=a, b=a)
        result = cdf_beta(3, cum=original.cum, ccum=original.ccum, x=0.5, b=a)
        if float(original.cum) > 0 and float(original.ccum) > 0:
            assert float(result.a) == pytest.approx(a, rel=1e-8)


def test_native_success_status_does_not_guarantee_a_quantile():
    case = next(
        c
        for c in FIXTURE["cases"]
        if c["input"][0] == 2 and c["input"][3:] == [0.01, 0.99, 1.0, 1.0]
    )
    assert case["status"] == 0
    assert case["result"][2] == 0.505
    assert sum(case["result"][2:4]) != pytest.approx(1)
    result = cdf_beta(2, cum=case["input"][1], ccum=case["input"][2], a=1, b=1)
    assert float(result.x) == pytest.approx(0.01)
    assert float(result.cx) == pytest.approx(0.99)


def test_native_exact_initial_shape_has_false_bound_status():
    failed = [c for c in FIXTURE["cases"] if c["status"] != 0]
    assert len(failed) == 11
    for case in failed:
        which, p, q, x, cx, a, b = case["input"]
        assert which == 4 and b == 5 and case["status"] == -50
        assert float(cdf_beta(4, cum=p, ccum=q, x=x, cx=cx, a=a).b) == pytest.approx(5, rel=1e-10)


@pytest.mark.parametrize("shape", [1e-10, 1e-6, 0.2, 1, 100, 1e6, 1e10])
def test_symmetric_midpoint_is_exact_and_reusable(shape):
    result = cdf_beta(x=0.5, a=shape, b=shape)
    assert float(result.cum) == float(result.ccum) == 0.5
    inverse = cdf_beta(3, x=0.5, b=shape, cum=result.cum, ccum=result.ccum)
    assert float(inverse.a) == pytest.approx(shape, rel=1e-8)


def test_large_shape_complements_can_be_reused_as_inputs():
    a = np.array([1e-10, 0.1, 1, 1e3, 1e6, 1e10])[:, None]
    b = np.array([1e-10, 0.1, 1, 1e3, 1e6, 1e10])[None, :]
    result = cdf_beta(x=0.5, a=a, b=b)
    np.testing.assert_array_equal(result.cum + result.ccum, np.ones((6, 6)))
    inverse = cdf_beta(2, a=a, b=b, cum=result.cum, ccum=result.ccum)
    assert np.all(np.isfinite(inverse.x))
