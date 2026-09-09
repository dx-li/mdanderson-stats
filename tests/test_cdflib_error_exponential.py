"""Public helper behavior, independent values and deliberate native repairs."""

import math
from decimal import Decimal, localcontext
from functools import partial

import numpy as np
import pytest
from test_cdflib_error_exponential_reference import FIXTURE, independent_error

from mdanderson_stats import erf, erfc1, esum, exparg


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_independent_values_and_native_contract(case):
    mode, ind, x = case["mode"], case["ind"], case["x"]
    if mode <= 2:
        expected = independent_error(x, scaled=mode == 2 and ind != 0, complement=mode == 2)
        function = partial(erf, x) if mode == 1 else partial(erfc1, ind, x)
    elif mode == 3:
        with localcontext() as ctx:
            ctx.prec = 100
            expected = float((Decimal(ind) + Decimal.from_float(x)).exp())
        function = partial(esum, ind, x)
    else:
        expected = case["result"]
        function = partial(exparg, ind)
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            function()
    else:
        np.testing.assert_allclose(function(), expected, rtol=2e-13, atol=5e-324)
        if mode == 2 and ind == 0 and 26.7 <= x <= 27.2:
            assert float(function()) > 0
        if mode == 3 and (ind, x) == (-745, 1.0):
            assert float(function()) == expected


@pytest.mark.parametrize("x", [26.0, np.nextafter(26.0, 27.0), 26.5, 26.65, 26.75, 27.1, 27.22])
def test_tail_switch_and_subnormal_values(x):
    np.testing.assert_allclose(
        erfc1(0, x), independent_error(x, complement=True), rtol=2e-13, atol=5e-324
    )


@pytest.mark.parametrize(
    "function", [erf, lambda x: erfc1(0, x), lambda x: erfc1(1, x), lambda x: esum(0, x)]
)
def test_array_ownership_empty_shape_and_signed_zero(function):
    x = np.array([[-0.0, 0.0], [-0.1, 0.1]])
    expected = np.array([[float(function(v)) for v in row] for row in x])
    result = function(x)
    x[:] = 99
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert function(np.empty((0, 3))).shape == (0, 3)
    assert function(0.0).shape == ()
    assert np.signbit(erf(-0.0))


def test_integer_coordinate_broadcasting_and_immutable_thresholds():
    x = np.array([[0.0, 1.0, 2.0]])
    flags = np.array([[0], [-2]])
    np.testing.assert_array_equal(erfc1(flags, x), [erfc1(0, x[0]), erfc1(1, x[0])])
    np.testing.assert_array_equal(esum(flags, x), [esum(0, x[0]), esum(-2, x[0])])
    limits = exparg(flags)
    flags[:] = 99
    assert limits.shape == (2, 1)
    with pytest.raises(ValueError):
        limits.setflags(write=True)
    assert exparg([]).shape == (0,)
    assert esum(np.empty((0, 1)), np.empty((1, 3))).shape == (0, 3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.5, -(2**31) - 1, 2**31])
@pytest.mark.parametrize("function", [exparg, lambda i: erfc1(i, 0), lambda i: esum(i, 0)])
def test_invalid_native_integer_domain(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("function", [erf, lambda x: erfc1(0, x), lambda x: esum(0, x)])
def test_nonfinite_coordinates_rejected(function, bad):
    with pytest.raises(ValueError):
        function(bad)


def test_nonbroadcastable_and_overflow_errors():
    for function in [erfc1, esum]:
        with pytest.raises(ValueError):
            function([0, 1], [0, 1, 2])
    with pytest.raises(ArithmeticError):
        erfc1([0, 1], [0, -27])
    with pytest.raises(ArithmeticError):
        esum([0, 0], [0, 710])


@pytest.mark.parametrize(
    "mu,x", [(700, 1e-8), (-700, -1e-8), (709, 0.5), (-744, 0.5), (2147483647, -2147483646.75)]
)
def test_combined_exponent_rounding(mu, x):
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float((Decimal(mu) + Decimal.from_float(x)).exp())
    np.testing.assert_allclose(esum(mu, x), expected, rtol=2e-13, atol=5e-324)


@pytest.mark.parametrize("x", np.linspace(27.20, 27.24, 17))
def test_last_representable_tails_round_only_at_final_product(x):
    assert float(erfc1(0, x)) == independent_error(float(x), complement=True)
