"""Gamma factor accuracy, source compatibility and real continuation."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_error_exponential_reference import decimal_pi
from test_cdflib_gamma_support_reference import positive_log_gamma
from test_cdflib_incomplete_gamma_reference import FIXTURE, independent, log_gamma

from mdanderson_stats import rcomp


def expected_factor(a, x):
    with localcontext() as ctx:
        ctx.prec = 800
        aa, xx = Decimal.from_float(float(a)), Decimal.from_float(float(x))
        if xx <= 0 or (aa <= 0 and aa == aa.to_integral_value()):
            return 0.0
        sign = 1
        if aa > 0:
            lg = log_gamma(aa)
        elif aa % 1 == Decimal("-.5"):
            lg = decimal_pi().ln() - positive_log_gamma(1 - aa)
            sign = -1 if int(-aa) % 2 == 0 else 1
        else:
            shifted = aa
            correction = Decimal(0)
            while shifted <= 0:
                correction -= abs(shifted).ln()
                sign = -sign
                shifted += 1
            lg = positive_log_gamma(shifted) + correction
        exponent = aa * xx.ln() - xx - lg
        if exponent > 1000:
            return math.copysign(math.inf, sign)
        if exponent < -1000:
            return math.copysign(0.0, sign)
        return sign * float(exponent.exp())


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 1])
def test_native_density_grid(case):
    a, x = case["a"], case["x"]
    expected = expected_factor(a, x) if a < 0 else independent(a, x)[0]
    value = float(rcomp(a, x))
    np.testing.assert_allclose(value, expected, rtol=5e-13, atol=5e-324)
    np.testing.assert_allclose(value, case["r"], rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "a,x",
    [
        (-0.5, 0.1),
        (-0.75, 1.0),
        (-1.5, 1.0),
        (-2.5, 2.0),
        (-175.5, 100.0),
        (-0.2, 5e-324),
        (-1e-100, 0.1),
        (-5e-324, 0.1),
        (5e-324, 0.1),
        (1.25, 744.0),
        (np.nextafter(1.25, 2.0), 744.0),
        (np.nextafter(8.0, 0.0), 8.0),
        (8.0, 8.0),
        (8.0, 5e-324),
        (8.0, np.finfo(float).max),
        (1e20, np.nextafter(1e20, 0.0)),
        (1e20, 1e20 + 1e11),
        (1e30, 1e30 + 4e16),
        (1e30, 1e30 - 4e16),
        (1e308, 1e308),
        (np.finfo(float).max, np.finfo(float).max),
        (1e308, np.nextafter(1e308, 0.0)),
    ],
)
def test_extreme_shapes_remainders_and_signed_continuation(a, x):
    expected = expected_factor(a, x)
    result = float(rcomp(a, x))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    assert np.signbit(result) == np.signbit(expected)


@pytest.mark.parametrize("a", [0.0, -1.0, -2.0, -1e308])
def test_reciprocal_gamma_zeros(a):
    assert float(rcomp(a, 1)) == 0


@pytest.mark.parametrize("a", [-175.5, -0.5, 0.0, 1.0, 1e308])
def test_source_nonpositive_coordinate_return(a):
    np.testing.assert_array_equal(rcomp(a, [-1, 0]), [0, 0])


def test_overflow_and_signed_underflow():
    with pytest.raises(ArithmeticError):
        rcomp(-175.5, 5e-324)
    result = float(rcomp(-0.5, 1e308))
    assert result == 0 and np.signbit(result)


@pytest.mark.parametrize("a,x", [(np.nan, 1), (1, np.inf), (-np.inf, 0)])
def test_nonfinite_inputs(a, x):
    with pytest.raises(ValueError):
        rcomp(a, x)


def test_broadcasting_and_owned_immutable_results():
    a, x = np.array([[0.5], [2.0]]), np.array([[1.0, 2.0, 3.0]])
    result = rcomp(a, x)
    expected = np.array([[expected_factor(float(v), float(w)) for w in x[0]] for v in a[:, 0]])
    a[:] = 99
    x[:] = 99
    np.testing.assert_allclose(result, expected, rtol=5e-13)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert rcomp(np.empty((0, 1)), np.empty((1, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        rcomp(np.ones(2), np.ones(3))
