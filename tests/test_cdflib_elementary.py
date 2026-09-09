"""Independent high-precision values and unchanged native helper evidence."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import alnrel, evaluate_polynomial, rexp, rlog, rlog1

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_elementary.json").read_text())
FUNCTIONS = {1: alnrel, 2: rexp, 3: rlog, 4: rlog1}


def independent(mode, x, a=()):
    with localcontext() as ctx:
        ctx.prec = 800
        z = Decimal.from_float(x)
        if mode == 1:
            return float((1 + z).ln())
        if mode == 2:
            return float(z.exp() - 1)
        if mode == 3:
            return float(z - 1 - z.ln())
        if mode == 4:
            return float(z - (1 + z).ln())
        return float(
            sum(
                Decimal.from_float(c) * z**i if i else Decimal.from_float(c)
                for i, c in enumerate(a)
            )
        )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_contract_and_independent_values(case):
    mode, x, a = case["mode"], case["x"], case["a"]
    if mode == 5 and not a:
        assert case["execution_outcome"] == "process_error"
        with pytest.raises(ValueError):
            evaluate_polynomial(a, x)
        return
    if (mode in (1, 4) and x <= -1) or (mode == 3 and x <= 0):
        with pytest.raises(ValueError):
            FUNCTIONS[mode](x)
        if mode == 4:
            assert case["result"] == -1
        return
    if mode == 2 and x >= 710:
        assert case["result"] == "inf"
        with pytest.raises(ArithmeticError):
            rexp(x)
        return
    result = evaluate_polynomial(a, x) if mode == 5 else FUNCTIONS[mode](x)
    expected = independent(mode, x, a)
    np.testing.assert_allclose(result, expected, rtol=4e-15, atol=np.nextafter(0.0, 1.0))
    assert case["execution_outcome"] == "completed"
    if mode == 4 and 0 < abs(x) < 1e-159:
        # The native square of x/2 can round away a representable remainder.
        np.testing.assert_allclose(
            case["result"], expected, rtol=1e-14, atol=2 * np.nextafter(0.0, 1.0)
        )
    else:
        np.testing.assert_allclose(
            case["result"], expected, rtol=4e-14, atol=np.nextafter(0.0, 1.0)
        )


@pytest.mark.parametrize(
    "x",
    [
        -0.125,
        -1e-5,
        -1e-10,
        -1e-100,
        -5e-162,
        -3e-162,
        0.0,
        3e-162,
        5e-162,
        1e-100,
        1e-10,
        1e-5,
        0.125,
    ],
)
def test_800_digit_remainder_near_zero(x):
    expected = independent(4, x)
    np.testing.assert_allclose(rlog1(x), expected, rtol=3e-15, atol=np.nextafter(0.0, 1.0))
    assert float(rlog1(x)) >= 0
    if abs(x) == 3e-162:
        assert float(rlog1(x)) == np.nextafter(0.0, 1.0)
        native = next(c for c in FIXTURE["cases"] if c["mode"] == 4 and c["x"] == x)
        assert native["result"] == 0


@pytest.mark.parametrize(
    "x", [np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0), 1 - 2**-30, 1 + 2**-30, 0.875, 1.125]
)
def test_800_digit_remainder_near_one(x):
    np.testing.assert_allclose(rlog(x), independent(3, x), rtol=3e-15, atol=0)


@pytest.mark.parametrize("function", [alnrel, rexp, rlog, rlog1])
def test_array_shape_ownership_and_empty(function):
    values = np.array([[0.1, 0.5], [1.0, 2.0]])
    result = function(values)
    expected = np.array([[float(function(x)) for x in row] for row in values])
    values[:] = 9
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert function([]).shape == (0,)


def test_polynomial_order_batch_and_precision():
    a = [1.0, -4.0, 6.0, -4.0, 1.0]
    x = np.array([[-2.0, 0.0], [0.5, 2.0]])
    np.testing.assert_array_equal(evaluate_polynomial(a, x), (x - 1) ** 4)
    assert evaluate_polynomial([2.0], []).shape == (0,)
    with pytest.raises(ValueError):
        evaluate_polynomial(a, x).setflags(write=True)
    with pytest.raises(ArithmeticError):
        evaluate_polynomial([1e308, 1e308], 2)


@pytest.mark.parametrize(
    "function,value",
    [
        (alnrel, -1),
        (alnrel, np.nan),
        (rexp, np.inf),
        (rlog, 0),
        (rlog, -1),
        (rlog1, -1),
        (rlog1, -2),
    ],
)
def test_invalid_domains(function, value):
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize("a,x", [([], 1), ([[1, 2]], 1), ([np.inf], 1), ([1], np.nan)])
def test_invalid_polynomial(a, x):
    with pytest.raises(ValueError):
        evaluate_polynomial(a, x)


@pytest.mark.parametrize("x", [-5e-324, 5e-324, -1e-320, 1e-320])
def test_subnormal_first_order_functions(x):
    assert float(alnrel(x)) == x
    assert float(rexp(x)) == x
    assert float(rlog1(x)) == 0


@pytest.mark.parametrize(
    "x",
    [
        np.nextafter(-0.125, -1.0),
        np.nextafter(-0.125, 0.0),
        np.nextafter(0.125, 0.0),
        np.nextafter(0.125, 1.0),
    ],
)
def test_series_switch_has_high_precision_agreement(x):
    np.testing.assert_allclose(rlog1(x), independent(4, x), rtol=4e-15, atol=0)
