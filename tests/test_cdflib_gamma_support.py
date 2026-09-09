"""Public gamma helper contracts, native repairs and independent identities."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_gamma_support_reference import FIXTURE, independent, positive_log_gamma

from mdanderson_stats import alngam, gam1, gamln, gamln1, gamma, log_gamma, psi

FUNCTIONS = {1: alngam, 2: gamln, 3: log_gamma, 4: gamln1, 5: gam1, 6: gamma, 7: psi}


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_reference_cases_against_independent_mathematics(case):
    mode, x = case["mode"], case["x"]
    invalid = (
        (mode in (2, 3) and x <= 0)
        or (mode in (1, 6, 7) and x <= 0 and x == math.floor(x))
        or (mode == 1 and x < 0 and int(-x) % 2 == 0)
    )
    if invalid:
        with pytest.raises(ValueError):
            FUNCTIONS[mode](x)
        return
    expected = independent(mode, x)
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            FUNCTIONS[mode](x)
        return
    result = float(FUNCTIONS[mode](x))
    np.testing.assert_allclose(result, expected, rtol=4e-13, atol=5e-324)
    if expected == 0:
        assert result == 0
    if mode == 6 and x in (-172.5, -175.5, -177.5):
        assert result != 0
        assert np.signbit(result) == np.signbit(expected)
    if mode in (4, 5) and abs(x) == 5e-324:
        assert result == expected


@pytest.mark.parametrize("function", list(FUNCTIONS.values()))
def test_array_shape_ownership_and_empty(function):
    a = np.array([[0.1, 0.5], [1.0, 1.2]])
    expected = np.array([[float(function(v)) for v in row] for row in a])
    result = function(a)
    a[:] = 99
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert function(np.empty((0, 3))).shape == (0, 3)
    assert function(0.5).shape == ()


@pytest.mark.parametrize("function", list(FUNCTIONS.values()))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_arguments(function, value):
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize(
    "function,values",
    [
        (gamln, [-1.5, 0]),
        (log_gamma, [-1.5, 0]),
        (alngam, [-0.5, -2, 0, -1e100]),
        (gamma, [-2, 0]),
        (psi, [-2, 0]),
        (gamln1, [np.nextafter(-0.2, -1), np.nextafter(1.25, 2)]),
        (gam1, [np.nextafter(-0.5, -1), np.nextafter(1.5, 2)]),
    ],
)
def test_domain_boundaries(function, values):
    for value in values:
        with pytest.raises(ValueError):
            function(value)


@pytest.mark.parametrize(
    "x",
    [
        0.8,
        np.nextafter(0.8, 0),
        np.nextafter(0.8, 1),
        2.25,
        np.nextafter(2.25, 3),
        1 - 2**-30,
        1 + 2**-30,
        np.nextafter(2.0, 1),
        np.nextafter(2.0, 3),
    ],
)
def test_log_gamma_switches_and_roots(x):
    for function in [alngam, gamln, log_gamma]:
        np.testing.assert_allclose(function(x), independent(2, x), rtol=5e-14, atol=0)


@pytest.mark.parametrize(
    "x", [-0.2, np.nextafter(0.6, 0), 0.6, np.nextafter(0.6, 1), -1e-5, 1e-5, 1e-100, 1.25]
)
def test_small_remainder_switches(x):
    for mode in [4, 5]:
        np.testing.assert_allclose(
            FUNCTIONS[mode](x), independent(mode, x), rtol=5e-14, atol=5e-324
        )


@pytest.mark.parametrize("x", [-10.75, -3.25, -1.25, -0.75, 0.25, 0.75, 1.25, 3.25, 10.75])
def test_gamma_and_digamma_recurrence_and_reflection(x):
    np.testing.assert_allclose(gamma(x + 1), x * gamma(x), rtol=1e-13, atol=0)
    np.testing.assert_allclose(psi(x + 1), psi(x) + 1 / x, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(gamma(x) * gamma(1 - x), math.pi / math.sin(math.pi * x), rtol=1e-13)
    np.testing.assert_allclose(
        psi(1 - x) - psi(x), math.pi / math.tan(math.pi * x), rtol=1e-13, atol=1e-14
    )


def test_signed_underflow_and_valid_negative_logarithms():
    result = gamma([-180.5, -181.5])
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [True, False])
    assert float(alngam(-177.5)) < -744
    assert float(psi(-2147483647.5)) > 20
    assert float(gamln(1e-309)) > 711
    assert float(alngam(1e-309)) > 711


@pytest.mark.parametrize(
    "function,value",
    [(gamln, 1e306), (alngam, 1e306), (log_gamma, 1e306), (gamma, 172), (psi, 1e-309)],
)
def test_batch_output_overflow(function, value):
    with pytest.raises(ArithmeticError):
        function([0.5, value])


@pytest.mark.parametrize(
    "x",
    [
        np.nextafter(-170.0, -171.0),
        np.nextafter(-170.0, -169.0),
        np.nextafter(-175.0, -176.0),
        np.nextafter(-175.0, -174.0),
        -1e-100,
        -0.001,
        -1.001,
        -1.999,
    ],
)
def test_negative_gamma_near_poles_by_decimal_recurrence(x):
    with localcontext() as ctx:
        ctx.prec = 100
        z = Decimal.from_float(float(x))
        product = Decimal(1)
        while z < 1:
            product *= z
            z += 1
        expected = positive_log_gamma(z).exp() / product
        np.testing.assert_allclose(gamma(x), float(expected), rtol=3e-13, atol=5e-324)
        if expected > 0:
            np.testing.assert_allclose(alngam(x), float(expected.ln()), rtol=3e-13, atol=1e-14)
        else:
            with pytest.raises(ValueError):
                alngam(x)
