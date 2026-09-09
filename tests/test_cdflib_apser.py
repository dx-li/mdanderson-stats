"""Independent upper beta integrals and exact source-domain boundaries."""

from fractions import Fraction

import numpy as np
import pytest
from test_cdflib_beta_series_reference import FIXTURE, independent

from mdanderson_stats import apser


def valid(a, b, x, eps):
    a, b, x, eps = map(Fraction.from_float, (a, b, x, eps))
    return a <= min(eps, eps * b) and b * x <= 1 and x <= Fraction(1, 2)


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 1])
def test_native_cases_against_beta_integrals(case):
    a, b, x, eps = (case[k] for k in ("a", "b", "x", "eps"))
    if eps <= 0:
        with pytest.raises(ValueError):
            apser(a, b, x, eps)
        return
    expected = independent(1, a, b, x)
    result = apser(a, b, x, eps)
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize(
    "a,eps",
    [
        (5e-324, 5e-15),
        (1e-309, 5e-15),
        (1e-20, 5e-15),
        (1e-18, 5e-15),
        (5e-15, 5e-15),
        (0.01, 0.1),
        (0.1, 1.0),
        (2.0, 10.0),
    ],
)
@pytest.mark.parametrize(
    "b", [5e-324, 1e-309, 1e-18, 0.1, 0.25, 0.5, 2.0, 8.0, 1e8, 1e14, 1e15, 1e20, 1e308]
)
@pytest.mark.parametrize("coordinate", [0, 1, 2, 3])
def test_shape_and_regime_grid(a, eps, b, coordinate):
    x = [0.0, 5e-324, min(0.5, 1 / b), min(0.1, 0.5 / b)][coordinate]
    if not valid(a, b, x, eps):
        with pytest.raises(ValueError):
            apser(a, b, x, eps)
        return
    expected = independent(1, a, b, x)
    result = float(apser(a, b, x, eps))
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize(
    "a,b",
    [(0.1, 0.1), (0.125, 0.125), (1e-18, 1e-18), (5e-324, 5e-324), (1e-20, 5e-324), (0.01, 0.2)],
)
@pytest.mark.parametrize("x", [5e-324, 0.1, 0.5])
def test_small_shapes_with_loose_domain(a, b, x):
    expected = independent(1, a, b, x)
    np.testing.assert_allclose(apser(a, b, x, 1e308), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [5e-324, 1e-18, 0.1, 0.5, 2.0, 10.0, 100.0])
@pytest.mark.parametrize("b", [1e15, 1e20, 1e308])
def test_large_companion_limit(a, b):
    x = 0.5 / b
    expected = independent(1, a, b, x)
    np.testing.assert_allclose(apser(a, b, x, 100.0), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("b", [0.1, 0.3, 0.7, 1e-309])
@pytest.mark.parametrize("a", [5e-324, 1e-20])
def test_inclusive_rounded_shape_boundary(a, b):
    eps = a / b
    if valid(a, b, 0.1, eps):
        np.testing.assert_allclose(
            apser(a, b, 0.1, eps), independent(1, a, b, 0.1), rtol=5e-13, atol=5e-324
        )
    else:
        with pytest.raises(ValueError):
            apser(a, b, 0.1, eps)


@pytest.mark.parametrize("b", [3.0, 10.0, 1e15, 1e20])
def test_exact_coordinate_product_boundary(b):
    for x in [np.nextafter(1 / b, 0), 1 / b, np.nextafter(1 / b, np.inf)]:
        if valid(1e-20, b, float(x), 5e-15):
            np.testing.assert_allclose(
                apser(1e-20, b, x), independent(1, 1e-20, b, float(x)), rtol=5e-13, atol=0
            )
        else:
            with pytest.raises(ValueError):
                apser(1e-20, b, x)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(a=0),
        dict(b=0),
        dict(a=np.inf),
        dict(x=-0.1),
        dict(x=0.6),
        dict(eps=0),
        dict(eps=-1),
        dict(eps=np.nan),
        dict(a=1e-10),
    ],
)
def test_invalid_inputs(kwargs):
    args = dict(a=1e-20, b=1.0, x=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        apser(**args)


def test_broadcast_and_immutable_ownership():
    b = np.array([[0.1], [2.0]])
    x = np.array([0.0, 0.1, 0.5])
    result = apser(1e-20, b, x)
    expected = [[independent(1, 1e-20, float(v), float(z)) for z in x] for v in b[:, 0]]
    b[:] = 99
    x[:] = 99
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=0)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert apser(1e-20, np.empty((0, 1)), np.empty((1, 3))).shape == (0, 3)


@pytest.mark.parametrize("a", [0.9e-18, 1.1e-18, 4e-15])
@pytest.mark.parametrize("b", [0.3, 2.0, 1e14])
def test_small_upper_tails_across_numerical_regimes(a, b):
    x = min(0.5, 0.8 / b)
    np.testing.assert_allclose(apser(a, b, x, 5e-14), independent(1, a, b, x), rtol=5e-13, atol=0)


@pytest.mark.parametrize("b", [1.0, 2.0, 1e308])
def test_unit_first_shape_exact_upper_tail(b):
    x = 0.5 / b
    np.testing.assert_allclose(apser(1.0, b, x, 1.0), independent(1, 1.0, b, x), rtol=2e-15)


def test_huge_shapes_with_overflowing_sum():
    assert apser(1e308, 1e308, 5e-309, 1e308) == independent(1, 1e308, 1e308, 5e-309) == 1


@pytest.mark.parametrize("a,b,eps", [(0.01, 0.5, 0.1), (5e-15, 2.0, 5e-15)])
@pytest.mark.parametrize(
    "x",
    [
        np.nextafter(np.finfo(float).tiny, 0),
        np.finfo(float).tiny,
        np.nextafter(np.finfo(float).tiny, np.inf),
    ],
)
def test_smallest_normal_coordinate_boundary(a, b, eps, x):
    np.testing.assert_allclose(
        apser(a, b, x, eps), independent(1, a, b, float(x)), rtol=5e-13, atol=0
    )
