"""Independent tails, source contracts and repaired native failures."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_gamma_factor import expected_factor
from test_cdflib_incomplete_gamma_reference import FIXTURE, independent

from mdanderson_stats import grat1, gratio, rcomp


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] in (2, 3)])
def test_native_grid_against_independent_tails(case):
    a, x, mode = case["a"], case["x"], case["mode"]
    invalid = a < 0 or x < 0 or (mode == 2 and a == x == 0)
    invalid |= mode == 3 and case["eps"] <= 0
    function = gratio if mode == 2 else grat1
    args = (a, x, case["ind"]) if mode == 2 else (a, x, case["r"], case["eps"])
    if invalid:
        with pytest.raises(ValueError):
            function(*args)
        return
    if a == x == 0:
        expected = (0.0, 1.0)
    else:
        _, p, q = independent(a, x)
        if mode == 3 and a > 0 and a != 0.5 and x >= 1.1:
            q *= case["factor"]
            p = 1 - q
        expected = (p, q)
    np.testing.assert_allclose(function(*args), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [1e-100, 1e-16, 1e-12, 1e-5, 0.1, 0.5, 1.0, 2.0, 20.0, 100.0])
@pytest.mark.parametrize("x", [1e-300, 0.01, 1.0999, 1.1, 1.1001, 5.0, 10.0, 100.0, 730.0, 744.0])
def test_tail_grid_switches_and_subnormals(a, x):
    _, p, q = independent(a, x)
    result = gratio(a, x)
    np.testing.assert_allclose(result, (p, q), rtol=5e-13, atol=5e-324)
    if a <= 1:
        np.testing.assert_allclose(grat1(a, x, rcomp(a, x)), (p, q), rtol=5e-13, atol=5e-324)
    assert abs(float(result[0] + result[1]) - 1) <= np.finfo(float).eps


@pytest.mark.parametrize("a", [5e-324, 1e-323, 1.5e-323, 1e-100, 0.1, 0.5])
@pytest.mark.parametrize("x", [5e-324, 0.1, 0.5, 1.0, 1.1])
def test_positive_inputs_are_not_zero_product_endpoints(a, x):
    expected = independent(a, x)[1:]
    for result in (gratio(a, x), grat1(a, x, rcomp(a, x))):
        np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
        for value, target in zip(result, expected, strict=True):
            if target > 0:
                assert float(value) > 0


def test_scaling_factor_branch_contract():
    r = rcomp(0.1, 1.1)
    p, q = grat1(0.1, 1.1, r)
    doubled = grat1(0.1, 1.1, 2 * r)
    assert float(doubled[1]) == 2 * float(q)
    assert float(doubled[0]) == 1 - 2 * float(q)
    assert float(p) == 1 - float(q)
    np.testing.assert_array_equal(grat1(0.1, 0.1, 0), grat1(0.1, 0.1, 999))
    np.testing.assert_array_equal(grat1(0.5, 2, 0), grat1(0.5, 2, 999))
    with pytest.raises(ArithmeticError):
        grat1(0.1, 1.1, 999)


@pytest.mark.parametrize("ind", [-(2**31), -1, 0, 1, 2, 2**31 - 1])
def test_accuracy_selectors_use_full_accuracy(ind):
    np.testing.assert_array_equal(gratio(20, 20, ind), gratio(20, 20))


@pytest.mark.parametrize("eps", [5e-324, 1e-100, 1e-3, 1e308])
def test_positive_tolerance_with_float64_accuracy_floor(eps):
    np.testing.assert_allclose(
        grat1(0.1, 2, rcomp(0.1, 2), eps), independent(0.1, 2.0)[1:], rtol=5e-13
    )


@pytest.mark.parametrize("ind", [0.5, 2**31, -(2**31) - 1, np.nan])
def test_invalid_selectors(ind):
    with pytest.raises(ValueError):
        gratio(1, 1, ind)


@pytest.mark.parametrize(
    "a,x,r,eps",
    [
        (-0.1, 1, 1, 1e-14),
        (1.1, 1, 1, 1e-14),
        (1, -1, 1, 1e-14),
        (1, 1, -1, 1e-14),
        (1, 1, 1, 0),
        (1, 1, np.inf, 1e-14),
        (1, 1, 1, np.nan),
    ],
)
def test_invalid_small_shape_inputs(a, x, r, eps):
    with pytest.raises(ValueError):
        grat1(a, x, r, eps)


@pytest.mark.parametrize("function", [gratio, lambda a, x: grat1(a, x, rcomp(a, x))])
def test_broadcasting_empty_and_immutable_results(function):
    a, x = np.array([[0.1], [0.5]]), np.array([[0.1, 1.1, 10]])
    p, q = function(a, x)
    expected = np.array([[independent(float(v), float(w))[1:] for w in x[0]] for v in a[:, 0]])
    a[:] = 99
    x[:] = 99
    np.testing.assert_allclose(p, expected[:, :, 0], rtol=5e-13)
    np.testing.assert_allclose(q, expected[:, :, 1], rtol=5e-13)
    for value in (p, q):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert function(np.empty((0, 1)), np.empty((1, 3)))[0].shape == (0, 3)


@pytest.mark.parametrize("a,x", [(200.0, 2.0), (10000.0, 6700.0)])
def test_lower_subnormal_tail_against_finite_integer_sum(a, x):
    expected = independent(a, x)[1]
    assert 0 < expected < np.finfo(float).tiny
    np.testing.assert_allclose(gratio(a, x)[0], expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [1e20, 1e30])
def test_large_shape_lower_subnormal_against_scaled_integral(a):
    x = a - 38.2 * math.sqrt(a)
    # Composite Simpson integration of the defining scaled integral, with
    # Decimal exponent evaluation to resolve 1-exp(-t)-t near zero. A 0.02
    # mesh has O(h**4) error well below one ulp of these subnormal tails.
    with localcontext() as ctx:
        ctx.prec = 80
        aa, xx = Decimal.from_float(a), Decimal.from_float(x)
        d = aa - xx
        values = []
        for k in range(2001):
            u = Decimal(k) / 50
            t = u / d
            exponent = -u + xx * (1 - (-t).exp() - t)
            weight = 1 if k in (0, 2000) else (4 if k % 2 else 2)
            values.append(weight * float(exponent.exp()))
        ratio = math.fsum(values) / 150 / float(d)
    expected = expected_factor(a, x) * ratio
    assert 0 < expected < np.finfo(float).tiny
    np.testing.assert_allclose(gratio(a, x)[0], expected, rtol=0, atol=5e-324)
