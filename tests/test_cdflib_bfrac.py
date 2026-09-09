"""Independent beta integrals for the reflected continued fraction."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_factors_reference import beta_cdf
from test_cdflib_beta_remaining_reference import FIXTURE, expected, integral

from mdanderson_stats import bfrac


def oracle(a, b, x, y=None):
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb, xx = map(Decimal.from_float, (a, b, x))
        yy = 1 - xx if y is None else Decimal.from_float(y)
        xx = xx if xx <= yy else 1 - yy
        if xx == 0 or xx == 1:
            return float(xx)
        if aa == bb and aa * (4 * xx * (1 - xx)).ln() < -800:
            return 0.0 if xx < Decimal(".5") else 1.0
        if a.is_integer() and b.is_integer() and a + b < 10000:
            return float(beta_cdf(aa, bb, xx, 1 - xx))
        return integral(aa, bb, xx)


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 2])
def test_native_cases_against_independent_integrals(case):
    np.testing.assert_allclose(
        bfrac(case["a"], case["b"], case["x"], case["y"], case["eps"]),
        expected(case),
        rtol=5e-13,
        atol=5e-324,
    )


@pytest.mark.parametrize("a", [1.0001, 1.5, 2.0, 15.0, 100.0])
@pytest.mark.parametrize("b", [1.0001, 1.5, 2.0, 15.0, 100.0])
@pytest.mark.parametrize("x", [0.0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0])
def test_integral_grid(a, b, x):
    target = oracle(a, b, x)
    result = bfrac(a, b, x)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("a,b", [(1000.0, 2000.0), (2000.0, 1000.0), (1000.0, 1000.0)])
@pytest.mark.parametrize("x", [0.01, 0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 0.9])
def test_large_shapes_against_positive_binomial_sum(a, b, x):
    np.testing.assert_allclose(bfrac(a, b, x), oracle(a, b, x), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a,b", [(1.5, 1e308), (15.0, 1e308), (1e308, 1.5), (1e308, 15.0)])
@pytest.mark.parametrize("z", [5e-324, 1e-310, 1e-308, 1e-307])
def test_extreme_shapes_with_explicit_complements(a, b, z):
    x, y = (z, 1.0) if a < b else (1.0, z)
    target = oracle(a, b, x, y)
    result = bfrac(a, b, x, y)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("a", [1e30, 1e100, 1e308])
@pytest.mark.parametrize("x", [0.1, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 0.9])
def test_huge_symmetric_shapes(a, x):
    np.testing.assert_allclose(bfrac(a, a, x), oracle(a, a, float(x)), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 2.0, 0.5),
        (2.0, 1.0, 0.5),
        (np.inf, 2.0, 0.5),
        (2.0, np.nan, 0.5),
        (2.0, 3.0, -0.1),
        (2.0, 3.0, 1.1),
        (2.0, 3.0, 0.5, 0.6),
        (2.0, 3.0, None),
        (2.0, 3.0, 0.5, None, 0.0),
        (2.0, 3.0, 0.5, None, np.inf),
    ],
)
def test_invalid_domains(args):
    with pytest.raises(ValueError):
        bfrac(*args)


def test_broadcasting_and_ownership():
    a = np.array([[2.0], [15.0]])
    x = np.array([0.1, 0.5, 0.9])
    result = bfrac(a, 3.0, x, eps=1e10)
    target = [[oracle(float(v), 3.0, float(z)) for z in x] for v in a[:, 0]]
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=0)
    a[:] = 30
    x[:] = 0
    assert np.all(result > 0)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    np.testing.assert_array_equal(bfrac(2.0, 3.0, None, [0.0, 0.5, 1.0]), [1.0, 0.6875, 0.0])


def test_large_batch_against_independent_integrals():
    points = np.array([0.01, 0.2, 0.5, 0.7, 0.99])
    x = np.tile(points, 1000)
    target = np.tile([oracle(20.0, 50.0, float(z)) for z in points], 1000)
    np.testing.assert_allclose(bfrac(20.0, 50.0, x), target, rtol=5e-13, atol=0)


@pytest.mark.parametrize("eps", [5e-324, 5e-15, 1e10])
@pytest.mark.parametrize(
    "a,b,x", [(1.0000000000000002, 1.5, 0.7), (20.0, 50.0, 0.5), (100.0, 100.0, 0.49)]
)
def test_tolerance_extremes(a, b, x, eps):
    np.testing.assert_allclose(bfrac(a, b, x, eps=eps), oracle(a, b, x), rtol=5e-13, atol=0)


@pytest.mark.parametrize("x", [np.nextafter(0.035, 0.0), 0.035, np.nextafter(0.035, 1.0)])
def test_power_series_transition(x):
    np.testing.assert_allclose(bfrac(2.0, 20.0, x), oracle(2.0, 20.0, float(x)), rtol=5e-13, atol=0)


@pytest.mark.parametrize("x", [np.nextafter(0.485, 0.0), 0.485, np.nextafter(0.485, 1.0)])
def test_asymptotic_transition(x):
    np.testing.assert_allclose(
        bfrac(1000.0, 1000.0, x), oracle(1000.0, 1000.0, float(x)), rtol=5e-13, atol=0
    )


@pytest.mark.parametrize("a,b", [(100.0, 1000.0), (15.0, 100.0), (2.0, 2.0)])
@pytest.mark.parametrize("x", [1e-3, 1e-5, 1e-160, 1e-162])
def test_far_and_subnormal_tails(a, b, x):
    target = oracle(a, b, x)
    result = bfrac(a, b, x)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("a", [1.5, 8.0, 15.0, 50.0, 100.0])
@pytest.mark.parametrize("scaled_x", [0.71, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
def test_large_companion_fraction_normalization(a, scaled_x):
    b = 1e308
    x = scaled_x / b
    target = oracle(a, b, x)
    np.testing.assert_allclose(bfrac(a, b, x), target, rtol=5e-13, atol=5e-324)
