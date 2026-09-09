"""Independent beta increments and accumulator semantics for bgrat."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_remaining_reference import FIXTURE, expected, integral

from mdanderson_stats import bgrat


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 3])
def test_native_cases_against_independent_increment(case):
    actual = bgrat(case["a"], case["b"], case["x"], case["y"], case["initial"], case["eps"])
    np.testing.assert_allclose(actual, expected(case), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [15.0, 100.0, 1e4, 1e12, 1e308])
@pytest.mark.parametrize("b", [5e-324, 1e-309, 1e-20, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("y", [0.0, 5e-324, 1e-310, 1e-12, 0.1, 0.5, 1.0])
def test_complementary_coordinate_grid(a, b, y):
    with localcontext() as ctx:
        ctx.prec = 800
        target = integral(Decimal.from_float(a), Decimal.from_float(b), 1 - Decimal.from_float(y))
    result = bgrat(a, b, None, y)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    if target > 0:
        assert result > 0


@pytest.mark.parametrize("w", [-1e308, -1.0, 0.0, 0.25, 1.0, 1e308])
def test_accumulator_is_added_without_probability_clipping(w):
    # b=1 gives the exact increment x**a, here 2**-15.
    target = w + 2**-15
    result = bgrat(15, 1, 0.5, w=w)
    np.testing.assert_allclose(result, target, rtol=2e-15, atol=0)
    if w < 0:
        assert result < 0
    if w >= 1:
        assert result >= 1


def test_broadcasting_ownership_and_loose_tolerance():
    x = np.array([0.0, 0.1, 0.5, 1.0])
    w = np.array([[-1.0], [0.25], [1.0]])
    result = bgrat(15, 1, x, w=w, eps=1e10)
    np.testing.assert_allclose(result, w + x**15, rtol=2e-15, atol=0)
    assert not result.flags.writeable
    x[:] = 0.5
    w[:] = 9
    np.testing.assert_array_equal(result[:, 0], [-1, 0.25, 1])
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 14.999},
        {"b": 0},
        {"b": 1.01},
        {"x": -0.1},
        {"y": 0.1},
        {"eps": 0},
        {"eps": -1},
        {"eps": np.inf},
        {"a": np.inf},
        {"b": np.nan},
        {"w": np.inf},
        {"w": np.nan},
    ],
)
def test_invalid_inputs(kwargs):
    inputs = dict(a=15, b=0.5, x=0.5)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        bgrat(**inputs)
