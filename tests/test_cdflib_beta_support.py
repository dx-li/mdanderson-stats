"""Independent beta and continuous-combinatorial values across the native domain."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_support_reference import FIXTURE, independent

from mdanderson_stats import betaln, log_beta, log_bicoef

FUNCTIONS = {3: betaln, 4: log_beta, 6: log_bicoef}


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] in FUNCTIONS])
def test_native_grid_against_independent_values(case):
    mode, a, b = case["mode"], case["a"], case["b"]
    invalid = (mode in (3, 4) and min(a, b) <= 0) or (
        mode == 6 and (a <= -1 or b <= -1 or b - a <= -1)
    )
    if invalid:
        with pytest.raises(ValueError):
            FUNCTIONS[mode](a, b)
        return
    expected = independent(mode, a, b)
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            FUNCTIONS[mode](a, b)
        return
    result = float(FUNCTIONS[mode](a, b))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if mode == 6 and 0 < a < 1e-90 and b == 1:
        assert result > 0
    if expected == 0:
        assert result == 0


@pytest.mark.parametrize(
    "a,b",
    [
        (1 - 2**-53, 1 + 2**-52),
        (0.9, 1.1),
        (0.5, 0.5),
        (7.99, 8.0),
        (8.0, 8.0),
        (8.0, np.finfo(float).max),
        (1e-309, 1e308),
        (1e308, 1e308),
        (1 + 2**-52, 1 + 2**-52),
    ],
)
def test_beta_symmetry_switches_and_wide_inputs(a, b):
    value = betaln(a, b)
    assert float(value) == float(betaln(b, a)) == float(log_beta(a, b))
    np.testing.assert_allclose(value, independent(3, float(a), float(b)), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "k,n",
    [
        (1.0, 5e-324),
        (1.0, 1e-100),
        (1.0, 1e308),
        (0.0, -0.5),
        (-0.5, -0.5),
        (5e-324, 1.0),
        (5e-324, 7.0),
        (1e-100, 7.0),
        (-1e-100, 7.0),
        (-1e-100, -0.5),
        (1e-100, -0.5),
        (1e-100, 1e308),
        (0.5e-100, 1e-100),
        (1e-100, 0.0),
        (-1e-100, 1e-100),
        (1.5e-162, 3e-162),
        (0.125, 0.25),
        (0.25, 0.0),
        (-0.25, 0.0),
        (5e-324, 0.25),
        (5e-324, 0.5),
        (1e-200, 0.0),
        (-0.125, 0.125),
        (0.125, np.nextafter(0.25, 1)),
        (0.3, 0.5),
        (-0.3, -0.7),
        (1.5, 1.0),
        (-0.5, 0.5),
        (-0.25, -0.5),
        (5e307, 1e308),
    ],
)
def test_fractional_endpoint_and_small_argument_binomial(k, n):
    expected = independent(6, float(k), float(n))
    result = float(log_bicoef(k, n))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected != 0:
        assert result != 0
        assert np.signbit(result) == np.signbit(expected)


@pytest.mark.parametrize("n,k", [(10, 0), (10, 1), (10, 5), (10, 10), (100, 50)])
def test_exact_integer_coefficients(n, k):
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float(Decimal(math.comb(n, k)).ln())
    np.testing.assert_allclose(log_bicoef(k, n), expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize(
    "function,args", [(betaln, (0.5, 2.0)), (log_beta, (0.5, 2.0)), (log_bicoef, (0.5, 2.0))]
)
def test_broadcasting_owned_immutable_outputs_and_empty(function, args):
    a = np.full((2, 1), args[0])
    b = np.full((1, 3), args[1])
    expected = np.full((2, 3), float(function(*args)))
    result = function(a, b)
    a[:] = 99
    b[:] = 99
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert function(np.empty((0, 1)), np.empty((1, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        function(np.ones(2), np.ones(3))


@pytest.mark.parametrize(
    "function,a,b",
    [
        (betaln, 0, 1),
        (betaln, 1, -1),
        (log_beta, np.inf, 1),
        (log_beta, 1, np.nan),
        (log_bicoef, -1, 1),
        (log_bicoef, 0, -1),
        (log_bicoef, 2, 1),
        (log_bicoef, np.inf, 1),
        (log_bicoef, 1, -5e-324),
    ],
)
def test_invalid_domains(function, a, b):
    with pytest.raises(ValueError):
        function(a, b)


def test_beta_output_overflow_is_explicit():
    with pytest.raises(ArithmeticError):
        betaln(np.finfo(float).max, np.finfo(float).max)


@pytest.mark.parametrize("d", [2**-52, 2**-40, 0.25])
def test_opposing_unit_shape_offsets_preserve_quadratic_beta_logarithm(d):
    a, b = 1 - d, 1 + d
    expected = independent(3, a, b)
    assert expected > 0
    np.testing.assert_allclose(betaln(a, b), expected, rtol=5e-14, atol=0)


@pytest.mark.parametrize("n", [-0.9, -0.99, -0.75, np.nextafter(-1.0, 0.0)])
def test_fractional_coefficient_next_to_right_shape_boundary(n):
    k = float(np.nextafter(n + 1, 0.0))
    expected = independent(6, k, float(n))
    result = float(log_bicoef(k, n))
    assert math.isfinite(result) and result < 0
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=0)
