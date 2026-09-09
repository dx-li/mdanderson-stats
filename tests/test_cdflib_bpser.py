"""Independent beta integrals throughout the source power-series domain."""

from fractions import Fraction

import numpy as np
import pytest
from test_cdflib_beta_series_reference import FIXTURE, independent

from mdanderson_stats import bpser


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 3])
def test_native_cases_against_independent_integrals(case):
    a, b, x, eps = (case[k] for k in ("a", "b", "x", "eps"))
    if eps <= 0:
        with pytest.raises(ValueError):
            bpser(a, b, x, eps)
        return
    expected = independent(3, a, b, x)
    np.testing.assert_allclose(bpser(a, b, x, eps), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "a", [5e-324, 1e-309, 1e-20, 0.01, 0.5, 1.0, 1.01, 2.0, 10.0, 100.0, 1e308]
)
@pytest.mark.parametrize("b", [5e-324, 1e-309, 0.25, 0.5, 1.0, 2.0, 8.0, 1e10, 1e308])
@pytest.mark.parametrize("coordinate", [5e-324, 0.1, 0.5])
def test_full_shape_grid(a, b, coordinate):
    x = coordinate / max(1.0, b)
    expected = independent(3, a, b, x)
    result = bpser(a, b, x)
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize(
    "a,b", [(0.01, 0.5), (0.1, 5e-324), (0.5, 1e-20), (2.0, 1e-20), (1e12, 1e-20)]
)
@pytest.mark.parametrize("x", [1 - 1e-12, np.nextafter(1.0, 0.0)])
def test_near_one_against_reflected_integral(a, b, x):
    expected = independent(1, b, a, 1 - x)
    np.testing.assert_allclose(bpser(a, b, x), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("b", [1.01, 2.0, 8.0, 1e10, 1e308])
def test_exact_product_domain_boundary(b):
    center = 0.7 / b
    for x in [np.nextafter(center, 0.0), center, np.nextafter(center, np.inf)]:
        valid = Fraction.from_float(b) * Fraction.from_float(x) <= Fraction.from_float(0.7)
        if valid:
            # Uniform first shape gives an independent closed form.
            expected = -np.expm1(b * np.log1p(-x))
            np.testing.assert_allclose(bpser(1, b, x), expected, rtol=5e-15)
        else:
            with pytest.raises(ValueError, match="b<=1"):
                bpser(1, b, x)


@pytest.mark.parametrize(
    "args",
    [
        (0, 1, 0.5),
        (1, 0, 0.5),
        (1, 1, -0.1),
        (1, 1, 1.1),
        (1, 2, 0.5),
        (1, 1, 0.5, 0),
        (1, 1, 0.5, -1),
        (np.inf, 1, 0.5),
        (1, np.nan, 0.5),
        (1, 1, np.nan),
        (1, 1, 0.5, np.inf),
    ],
)
def test_invalid_domains(args):
    with pytest.raises(ValueError):
        bpser(*args)


def test_broadcast_ownership_endpoints_and_loose_tolerance():
    x = np.array([0.0, 0.1, 0.5, 1.0])
    a = np.array([[0.01], [1.0], [2.0]])
    result = bpser(a, 0.5, x, 1e10)
    expected = [[independent(3, float(v), 0.5, float(z)) for z in x] for v in a[:, 0]]
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    assert not result.flags.writeable
    x[:] = 0.2
    np.testing.assert_array_equal(result[:, [0, 3]], [[0, 1]] * 3)
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize("a", [0.01, 0.5, 2.0])
@pytest.mark.parametrize("b,x", [(1.01, 0.69), (1.1, 0.63), (1.2, 0.58)])
def test_signed_series_above_half_against_reflected_integral(a, b, x):
    expected = independent(1, b, a, 1 - x)
    np.testing.assert_allclose(bpser(a, b, x), expected, rtol=5e-13, atol=0)


@pytest.mark.parametrize("a", [7.9, 10.0, 100.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_large_companion_transition(a, scale):
    b = scale * a * a * 1e14
    x = 0.5 / b
    expected = independent(3, a, b, x)
    np.testing.assert_allclose(bpser(a, b, x), expected, rtol=5e-13, atol=5e-324)
