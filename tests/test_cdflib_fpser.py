"""Independent beta integrals, source bounds, and tiny-shape regressions."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from test_cdflib_beta_series_reference import FIXTURE, independent

from mdanderson_stats import fpser


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 2])
def test_native_cases_against_independent_integrals(case):
    a, b, x, eps = (case[k] for k in ("a", "b", "x", "eps"))
    if eps <= 0:
        with pytest.raises(ValueError):
            fpser(a, b, x, eps)
        return
    expected = independent(2, a, b, x)
    result = fpser(a, b, x, eps)
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize("a", [5e-324, 1e-309, 1e-20, 0.1, 0.5, 2.0, 100.0, 1e308])
@pytest.mark.parametrize("b", [5e-324, 1e-309, 1e-20])
@pytest.mark.parametrize("x", [0.0, 5e-324, 1e-308, 0.1, 0.5])
def test_positive_shape_grid(a, b, x):
    eps = 2.0 if a == b == 5e-324 else 5e-15
    valid = Fraction.from_float(b) < Fraction.from_float(eps) * min(1, Fraction.from_float(a))
    if not valid:
        with pytest.raises(ValueError):
            fpser(a, b, x, eps)
        return
    expected = independent(2, a, b, x)
    result = fpser(a, b, x, eps)
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize("a", [0.1, 0.3, 0.7, 1e-309])
@pytest.mark.parametrize("b", [5e-324, 1e-20])
def test_rounded_strict_domain_boundary(a, b):
    eps = b / a
    valid = Fraction.from_float(b) < Fraction.from_float(eps) * Fraction.from_float(a)
    if valid:
        np.testing.assert_allclose(
            fpser(a, b, 0.5, eps), independent(2, a, b, 0.5), rtol=5e-13, atol=5e-324
        )
    else:
        with pytest.raises(ValueError):
            fpser(a, b, 0.5, eps)


@pytest.mark.parametrize("eps", [0.1, 2.0, 10.0])
def test_loose_tolerance_retains_full_beta_normalization(eps):
    a, b = 0.5, eps / 10
    expected = independent(2, a, b, 0.5)
    np.testing.assert_allclose(fpser(a, b, 0.5, eps), expected, rtol=5e-13)


def test_large_tolerance_general_beta_integral():
    np.testing.assert_allclose(fpser(2, 3, 0.1, 10), 0.0523, rtol=2e-14)
    np.testing.assert_allclose(fpser(2, 1, 0.1, 10), 0.01, rtol=2e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(a=0),
        dict(a=-1),
        dict(b=0),
        dict(b=np.inf),
        dict(x=-0.1),
        dict(x=0.6),
        dict(eps=0),
        dict(eps=-1),
        dict(eps=np.inf),
        dict(b=5e-15),
    ],
)
def test_invalid_inputs(kwargs):
    args = dict(a=1, b=1e-20, x=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        fpser(**args)


def test_broadcast_and_immutable_ownership():
    a = np.array([[0.1], [2.0]])
    x = np.array([0.0, 0.1, 0.5])
    result = fpser(a, 1e-20, x)
    expected = [[independent(2, float(v), 1e-20, float(z)) for z in x] for v in a[:, 0]]
    a[:] = 99
    x[:] = 99
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=0)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert fpser(np.empty((0, 1)), 1e-20, np.empty((1, 3))).shape == (0, 3)


def test_subnormal_result_matches_exact_unit_shape_identity():
    with localcontext() as ctx:
        ctx.prec = 800
        b, x = Decimal.from_float(1e-15), Decimal.from_float(1e-308)
        expected = float(1 - (b * (1 - x).ln()).exp())
    assert float(fpser(1, 1e-15, 1e-308)) == expected == 1e-323


@pytest.mark.parametrize("a", [0.99, 1.01, 10.0, 1000.0])
@pytest.mark.parametrize("power_log", [700.0, 710.0, 744.0])
def test_subnormal_integrals_without_unit_shape_shortcut(a, power_log):
    x = float(np.exp(-power_log / a))
    expected = independent(2, a, 1e-15, x)
    result = float(fpser(a, 1e-15, x))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0
