"""Independent beta integrals retain the source displacement coordinate."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_factors_reference import beta_cdf, log_beta
from test_cdflib_beta_remaining_reference import FIXTURE, expected, integral

from mdanderson_stats import basym


def oracle(a, b, lam):
    with localcontext() as ctx:
        ctx.prec = 800
        a, b, lam = map(Decimal.from_float, (a, b, lam))
        return integral(a, b, (a - lam) / (a + b))


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 1])
def test_native_cases_against_independent_integrals(case):
    result = basym(case["a"], case["b"], case["lam"], case["eps"])
    np.testing.assert_allclose(result, expected(case), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [15.0, 20.0, 50.0, 100.0, 200.0])
@pytest.mark.parametrize("b", [15.0, 20.0, 50.0, 100.0, 200.0])
@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.1, 0.5, 0.9])
def test_shape_and_displacement_grid(a, b, fraction):
    lam = a * fraction
    target = oracle(a, b, lam)
    result = basym(a, b, lam)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("a", [15.0, 50.0, 200.0])
@pytest.mark.parametrize("b", [15.0, 50.0, 200.0, 1e308])
def test_last_representable_displacement_before_endpoint(a, b):
    lam = float(np.nextafter(a, 0.0))
    target = oracle(a, b, lam)
    result = basym(a, b, lam)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("a", [1e30, 1e100, 1e308])
@pytest.mark.parametrize("distance", [1e-14, 1e-8, 0.1, 1.0])
def test_huge_symmetric_shapes_preserve_displacement(a, distance):
    lam = float(np.sqrt(a) * distance)
    target = oracle(a, a, lam)
    result = basym(a, a, lam)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=0)
    assert result < 0.5


@pytest.mark.parametrize(
    "a,b,lam",
    [
        (15.0, 1e308, 0.0),
        (15.0, 1e308, 7.5),
        (1e308, 15.0, 0.0),
        (1e308, 15.0, 25.0),
        (1e308, 15.0, 65.0),
    ],
)
def test_unequal_extreme_shapes(a, b, lam):
    np.testing.assert_allclose(basym(a, b, lam), oracle(a, b, lam), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "args",
    [
        (14.0, 15.0, 0.0),
        (15.0, 14.0, 0.0),
        (15.0, 15.0, -1.0),
        (15.0, 15.0, 16.0),
        (np.inf, 15.0, 0.0),
        (15.0, np.nan, 0.0),
        (15.0, 15.0, np.inf),
        (15.0, 15.0, 0.0, 0.0),
        (15.0, 15.0, 0.0, np.nan),
    ],
)
def test_invalid_domains(args):
    with pytest.raises(ValueError):
        basym(*args)


def test_broadcasting_exact_boundaries_and_ownership():
    a = np.array([[15.0], [1e308]])
    lam = np.array([0.0, 1.0, 15.0])
    result = basym(a, 15.0, lam, 1e10)
    target = [
        [oracle(float(v), 15.0, float(displacement)) for displacement in lam] for v in a[:, 0]
    ]
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    a[:] = 50
    lam[:] = 0
    assert result[0, 0] == 0.5 and result[0, 2] == 0
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize("a,b", [(1000.0, 1000.0), (1000.0, 2000.0), (2000.0, 1000.0)])
@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.5, 0.9])
def test_large_integer_shapes_against_positive_binomial_sum(a, b, fraction):
    lam = a * fraction
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb, ll = map(Decimal.from_float, (a, b, lam))
        x = (aa - ll) / (aa + bb)
        target = float(beta_cdf(aa, bb, x, 1 - x))
    np.testing.assert_allclose(basym(a, b, lam), target, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [15.0, 50.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_large_companion_endpoint_scales(a, scale):
    b = scale * a * a * 1e14
    lam = float(np.nextafter(a, 0.0))
    np.testing.assert_allclose(basym(a, b, lam), oracle(a, b, lam), rtol=5e-13, atol=5e-324)


def test_large_batch_against_independent_integral_values():
    points = np.array([0.0, 1.0, 2.0, 3.0, 15.0])
    lam = np.tile(points, 1000)
    target = np.tile([oracle(15.0, 15.0, float(v)) for v in points], 1000)
    np.testing.assert_allclose(basym(15.0, 15.0, lam), target, rtol=5e-13, atol=0)


@pytest.mark.parametrize("eps", [5e-324, 5e-15, 1e10])
@pytest.mark.parametrize("a,b,lam", [(50.0, 15.0, 25.0), (1e308, 1e308, 1e146), (15.0, 50.0, 0.0)])
def test_tolerance_floor_and_cap_preserve_integral(a, b, lam, eps):
    np.testing.assert_allclose(basym(a, b, lam, eps), oracle(a, b, lam), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("lam", [np.nextafter(1000.0, 0.0), 1000.0, np.nextafter(1000.0, np.inf)])
def test_weighted_logarithm_transition_against_positive_sum(lam):
    with localcontext() as ctx:
        ctx.prec = 800
        a = Decimal(2000)
        x = (a - Decimal.from_float(float(lam))) / (2 * a)
        target = float(beta_cdf(a, a, x, 1 - x))
    np.testing.assert_allclose(basym(2000.0, 2000.0, lam), target, rtol=5e-13, atol=5e-324)


def positive_integral(a, b, x):
    term = total = Decimal(1)
    for n in range(1, 10000):
        term *= (a + b + n - 1) * x / (a + n)
        total += term
        if term < total * Decimal("1e-750"):
            break
    else:
        raise AssertionError("independent positive integral did not converge")
    return (a * x.ln() + b * (1 - x).ln() - a.ln() - log_beta(a, b)).exp() * total


@pytest.mark.parametrize(
    "a", [float(np.nextafter(1000.0, 0.0)), 1000.0, float(np.nextafter(1000.0, np.inf))]
)
def test_large_companion_far_tail_against_positive_integral(a):
    b = 1e308
    lam = 0.75 * a
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb, ll = map(Decimal.from_float, (a, b, lam))
        x = (aa - ll) / (aa + bb)
        target = float(positive_integral(aa, bb, x))
    np.testing.assert_allclose(basym(a, b, lam), target, rtol=5e-13, atol=0)


@pytest.mark.parametrize("a,b,x", [(2, 3, 0.1), (50, 15, 0.3)])
def test_positive_integral_reference_against_binomial_sum(a, b, x):
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb, xx = Decimal(a), Decimal(b), Decimal.from_float(x)
        np.testing.assert_allclose(
            float(positive_integral(aa, bb, xx)), float(beta_cdf(aa, bb, xx, 1 - xx)), rtol=2e-15
        )


@pytest.mark.parametrize("distance", [26.0, 26.6, 27.0, 27.2])
def test_extreme_symmetric_tails_retain_subnormal_probabilities(distance):
    a = 1e308
    lam = float(distance * np.sqrt(a))
    target = oracle(a, a, lam)
    result = basym(a, a, lam)
    assert target > 0 and result > 0
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
