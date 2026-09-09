"""Independent incomplete-beta integrals and shape-shift identities."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_factors_reference import FIXTURE, independent

from mdanderson_stats import bup


def integer_b_difference(a, b, x, y, n):
    # Integrating the beta density by parts b-1 times gives this finite
    # polynomial for I_x(a,b), for any positive real a and integer b.
    with localcontext() as ctx:
        ctx.prec = 800
        aa = Decimal.from_float(float(a))
        if x <= y:
            xx = Decimal.from_float(float(x))
            yy = 1 - xx
        else:
            yy = Decimal.from_float(float(y))
            xx = 1 - yy
        if xx == 0 or yy == 0:
            return 0.0

        def integral(shape):
            term = total = Decimal(1)
            for j in range(1, b):
                term *= (shape + j - 1) * yy / j
                total += term
            return (shape * xx.ln()).exp() * total

        return float(integral(aa) - integral(aa + int(n)))


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 3])
def test_native_shift_cases_against_beta_integrals(case):
    a, b, x, y, n, eps = (case[k] for k in ("a", "b", "x", "y", "n", "eps"))
    if n <= 0 or eps <= 0:
        with pytest.raises(ValueError):
            bup(a, b, x, y, n, eps)
        return
    expected = independent(3, a, b, x, y, n=n)
    result = bup(a, b, x, y, n, eps)
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize("a", [0.1, 1.0, 2.0, 100.0, 1e8, 1e20, 1e308])
@pytest.mark.parametrize("b", [1, 2, 3, 8])
@pytest.mark.parametrize("n", [1, 10, 10000, 2**31 - 1])
@pytest.mark.parametrize("y", [5e-324, 1e-100, 1e-12, 0.1, 0.5])
def test_real_shape_integer_companion_finite_integral(a, b, n, y):
    expected = integer_b_difference(a, b, 1 - y, y, n)
    result = float(bup(a, b, 1 - y, y, n))
    np.testing.assert_allclose(result, expected, rtol=3e-12, atol=5e-324)
    if expected > 0:
        assert result > 0


@pytest.mark.parametrize("a,b", [(5e-324, 5e-324), (1e-309, 1e-309), (0.1, 0.2), (0.5, 0.5)])
@pytest.mark.parametrize("n", [1, 2, 100, 10000])
def test_fractional_shapes_against_integral_series(a, b, n):
    expected = independent(3, float(a), float(b), 0.5, 0.5, n=n)
    np.testing.assert_allclose(bup(a, b, 0.5, 0.5, n), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [1e100, 1e308])
@pytest.mark.parametrize("n", [1, 100, 2**31 - 1])
def test_huge_center_retains_unrepresentable_shape_increment(a, n):
    expected = n / np.sqrt(a) / np.sqrt(4 * np.pi)
    np.testing.assert_allclose(bup(a, a, 0.5, 0.5, n), expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(a=0),
        dict(b=-1),
        dict(n=0),
        dict(n=-1),
        dict(n=0.5),
        dict(n=2**31),
        dict(eps=0),
        dict(eps=-1),
        dict(eps=np.inf),
        dict(x=0.1, y=0.8),
    ],
)
def test_invalid_inputs(kwargs):
    args = dict(a=1, b=1, x=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        bup(**args)


def test_broadcast_and_immutable_owned_results():
    a = np.array([[0.5], [2.0]])
    n = np.array([1, 2, 10])
    result = bup(a, 2, 0.5, n=n)
    expected = [[integer_b_difference(float(v), 2, 0.5, 0.5, int(k)) for k in n] for v in a[:, 0]]
    a[:] = 99
    n[:] = 99
    np.testing.assert_allclose(result, expected, rtol=1e-13)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert bup(np.empty((0, 1)), 2, np.empty((1, 3))).shape == (0, 3)
    assert bup(2, 3, None, 0.5) == bup(2, 3, 0.5)
    assert bup(2, 3, 0) == bup(2, 3, 1) == 0


@pytest.mark.parametrize(
    "a,b,y",
    [
        (1e12, 2, 1e-12),
        (1e12, 8, 1e-12),
        (1e10, 8, 1e-12),
        (1e-100, 2, 1e-12),
        (1e12, 8, 1e-15),
        (1e20, 8, 1e-20),
    ],
)
def test_large_shift_blocks_against_finite_integral(a, b, y):
    expected = integer_b_difference(a, b, 1 - y, y, 2**31 - 1)
    np.testing.assert_allclose(bup(a, b, 1 - y, y, 2**31 - 1), expected, rtol=5e-13, atol=0)


@pytest.mark.parametrize(
    "a,b,n", [(1e12, 1e-309, 2**31 - 1), (1e20, 1e-309, 2**31 - 1), (1e12, 1e-309, 10000)]
)
def test_tiny_companion_is_not_rounded_before_finite_sum(a, b, n):
    from test_cdflib_gamma_support_reference import positive_psi

    with localcontext() as ctx:
        ctx.prec = 100
        aa, bb = Decimal.from_float(a), Decimal.from_float(b)
        # At y=1e-100, the relative corrections from b and a*y are
        # negligible; the first-order b coefficient is a digamma difference.
        expected = float(bb * (positive_psi(aa + n) - positive_psi(aa)))
    np.testing.assert_allclose(bup(a, b, 1, 1e-100, n), expected, rtol=5e-13, atol=5e-324)
