"""Independent beta factors, scaled extremes and compensated center offsets."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_factors_reference import FIXTURE, independent
from test_cdflib_gamma_support_reference import positive_log_gamma

from mdanderson_stats import brcmp1, brcomp


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] in (1, 2)])
def test_native_factor_grid_against_independent_values(case):
    a, b, x, y = (case[k] for k in ("a", "b", "x", "y"))
    function = brcomp if case["mode"] == 1 else brcmp1
    args = (a, b, x, y) if case["mode"] == 1 else (case["mu"], a, b, x, y)
    if abs(x + y - 1) > 1e-15:
        with pytest.raises(ValueError):
            function(*args)
        return
    if a <= 0:
        expected = 0.0 if a == 0 else a * x**a * y
    else:
        expected = independent(case["mode"], a, b, x, y, case["mu"])
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            function(*args)
        return
    result = float(function(*args))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize("a,b", [(1e20, 2e20), (1e30, 2e30), (1e30, 1e30), (8.0, 1e308)])
@pytest.mark.parametrize("offset", [-3e-15, 0.0, 3e-15])
def test_large_shape_offsets_preserve_small_coordinate(a, b, offset):
    x = a / b / (1 + a / b) + offset
    if x <= 0:
        with pytest.raises(ValueError):
            brcomp(a, b, x, 1 - x)
        return
    y = 1 - x
    expected = independent(1, a, b, x, y)
    np.testing.assert_allclose(brcomp(a, b, x, y), expected, rtol=5e-13, atol=5e-324)
    assert float(brcomp(a, b, x, y)) == float(brcomp(b, a, y, x))


@pytest.mark.parametrize("mu", [-1401, -1000, -744, 0, 700, 1000, 1401])
@pytest.mark.parametrize("a", [5e-324, 1e-309, 1e-100, 1e100, 1e300])
def test_complete_scaling_across_exponential_switches(mu, a):
    expected = independent(2, a, a, 0.5, 0.5, mu)
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            brcmp1(mu, a, a, 0.5, 0.5)
    else:
        result = float(brcmp1(mu, a, a, 0.5, 0.5))
        np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
        if expected > 0:
            assert result > 0


@pytest.mark.parametrize("a,b,x,y", [(9.0, 15.0, 5e-324, 1.0), (15.0, 9.0, 1.0, 5e-324)])
def test_scaled_subnormal_coordinate_avoids_rounded_quotient(a, b, x, y):
    expected = independent(2, a, b, x, y, 6000)
    assert 0 < expected < 1
    np.testing.assert_allclose(brcmp1(6000, a, b, x, y), expected, rtol=3e-12, atol=0)


def log_abs_gamma(a):
    sign = 1
    correction = Decimal(0)
    while a <= 0:
        correction -= abs(a).ln()
        sign = -sign
        a += 1
    return positive_log_gamma(a) + correction, sign


@pytest.mark.parametrize(
    "a,b",
    [
        (-0.5, 1.0),
        (-0.75, 2.0),
        (-1.5, 3.0),
        (-0.1, -0.1),
        (-0.2, 0.1),
        (-1.5, np.nextafter(-0.5, 0.0)),
        (-10.5, np.nextafter(0.5, 1.0)),
        (-10.5, np.nextafter(0.5, 0.0)),
        (-10.5, 0.5 + 1e-15),
    ],
)
def test_gamma_defined_real_domain(a, b):
    with localcontext() as ctx:
        ctx.prec = 100
        aa, bb = Decimal.from_float(a), Decimal.from_float(b)
        ga, sa = log_abs_gamma(aa)
        gb, sb = log_abs_gamma(bb)
        gs, ss = log_abs_gamma(aa + bb)
        expected = sa * sb * ss * float(((aa + bb) * Decimal(".5").ln() + gs - ga - gb).exp())
    np.testing.assert_allclose(brcomp(a, b, 0.5, 0.5), expected, rtol=5e-13)


def test_negative_shape_with_huge_positive_companion():
    with localcontext() as ctx:
        ctx.prec = 800
        a, b = Decimal("-.5"), Decimal.from_float(1e308)
        x = Decimal.from_float(5e-324)
        ga, sa = log_abs_gamma(a)
        gb, sb = log_abs_gamma(b)
        gs, ss = log_abs_gamma(a + b)
        expected = sa * sb * ss * float((a * x.ln() + b * (1 - x).ln() + gs - ga - gb).exp())
    np.testing.assert_allclose(brcomp(-0.5, 1e308, 5e-324, 1.0), expected, rtol=5e-13)
    assert float(brcmp1(1, -5e-324, 1, 0.25, 0.75)) == -1e-323


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, 0.0)])
def test_zero_shape_with_positive_companion(a, b):
    assert float(brcmp1(1000, a, b, 0.5, 0.5)) == 0


@pytest.mark.parametrize(
    "a,b", [(0.0, 0.0), (-1.0, 2.0), (0.5, -0.5), (-0.5, -0.5), (-1e308, -1e308)]
)
def test_gamma_poles_are_explicit(a, b):
    with pytest.raises(ValueError):
        brcomp(a, b, 0.5, 0.5)


@pytest.mark.parametrize("mu", [0.5, 2**31, -(2**31) - 1, np.inf])
def test_invalid_scale(mu):
    with pytest.raises(ValueError):
        brcmp1(mu, 1, 1, 0.5, 0.5)


def test_coordinate_completion_broadcasting_and_immutable_ownership():
    a = np.array([[0.5], [8.0]])
    x = np.array([[0.1, 0.5, 0.9]])
    result = brcomp(a, 2, x)
    expected = np.array(
        [[independent(1, float(v), 2.0, float(w), float(1 - w)) for w in x[0]] for v in a[:, 0]]
    )
    a[:] = 99
    x[:] = 99
    np.testing.assert_allclose(result, expected, rtol=5e-13)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert brcomp(np.empty((0, 1)), 2, np.empty((1, 3))).shape == (0, 3)
    assert float(brcomp(1, 1, None, 0.25)) == float(brcomp(1, 1, 0.75, 0.25))


@pytest.mark.parametrize("a", [8.0, 15.0, 50.0, 100.0])
@pytest.mark.parametrize("scaled_x", [0.01, 0.71, 1.0, 2.0, 5.0, 10.0, 20.0])
@pytest.mark.parametrize("reflected", [False, True])
def test_extreme_companion_normalization_avoids_log_cancellation(a, scaled_x, reflected):
    b = 1e308
    x, y = scaled_x / b, 1.0
    if reflected:
        a, b, x, y = b, a, y, x
    expected = independent(1, a, b, x, y)
    np.testing.assert_allclose(brcomp(a, b, x, y), expected, rtol=5e-13, atol=5e-324)
