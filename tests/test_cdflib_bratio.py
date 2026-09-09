"""Independent paired beta integrals including zero-shape contracts."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_remaining_reference import ERRORS, FIXTURE
from test_cdflib_bfrac import oracle

from mdanderson_stats import bratio


def integer_companion_tails(a, b, x, y):
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb, xx, yy = map(Decimal.from_float, (a, b, x, y))
        xx, yy = (xx, 1 - xx) if xx <= yy else (1 - yy, yy)
        if xx == 0:
            return 0.0, 1.0
        if yy == 0:
            return 1.0, 0.0
        term = total = Decimal(1)
        for k in range(1, int(bb)):
            term *= (aa + k - 1) * yy / k
            total += term
        logp = aa * xx.ln() + total.ln()
        value = Decimal(0) if logp < -800 else logp.exp()
        return float(value), float(1 - value)


def tails(a, b, x, y=None):
    y = 1 - x if y is None else y
    # A finite positive identity handles an enormous shape with an integer
    # companion without constructing an enormous binomial sum.
    if a >= 1e100 and b.is_integer() and 0 < b <= 100:
        return integer_companion_tails(a, b, x, y)
    if b >= 1e100 and a.is_integer() and 0 < a <= 100:
        q, p = integer_companion_tails(b, a, y, x)
        return p, q
    return oracle(a, b, x, y), oracle(b, a, y, x)


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] == 4])
def test_native_contract_and_independent_tails(case):
    args = tuple(case[k] for k in ("a", "b", "x", "y"))
    if args in ERRORS:
        with pytest.raises(ValueError, match=f"IERR={ERRORS[args]}"):
            bratio(*args)
        return
    a, b, x, y = args
    np.testing.assert_allclose(bratio(*args), tails(a, b, x, y), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [1e-20, 0.01, 0.5, 1.0, 1.5, 2.0, 15.0, 100.0])
@pytest.mark.parametrize("b", [1e-20, 0.01, 0.5, 1.0, 1.5, 2.0, 15.0, 100.0])
@pytest.mark.parametrize("x", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_both_tails_against_integral_grid(a, b, x):
    result = bratio(a, b, x)
    target = tails(a, b, x)
    np.testing.assert_allclose(result, target, rtol=5e-13, atol=5e-324)
    np.testing.assert_equal(result[0] + result[1], 1.0)
    for value, expected in zip(result, target, strict=True):
        if expected > 0:
            assert value > 0


@pytest.mark.parametrize("shape", [5e-324, 0.5, 1.0, 1e308])
@pytest.mark.parametrize("x", [5e-324, 0.1, 0.5, 1.0])
def test_zero_shape_limits(shape, x):
    np.testing.assert_array_equal(bratio(0.0, shape, x), [1.0, 0.0])
    np.testing.assert_array_equal(bratio(shape, 0.0, None, x), [0.0, 1.0])


@pytest.mark.parametrize("a,b", [(1.5, 1e308), (15.0, 1e308), (1e308, 1.5), (1e308, 15.0)])
@pytest.mark.parametrize("z", [5e-324, 1e-310, 1e-308, 1e-307])
def test_explicit_extreme_complements(a, b, z):
    x, y = (z, 1.0) if a < b else (1.0, z)
    np.testing.assert_allclose(bratio(a, b, x, y), tails(a, b, x, y), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a,b", [(5e-324, 5e-324), (1e-309, 5e-324), (5e-324, 1e-309)])
@pytest.mark.parametrize("x", [5e-324, 0.1, 0.5, 0.9])
def test_subnormal_shapes_retain_both_tails(a, b, x):
    np.testing.assert_allclose(bratio(a, b, x), tails(a, b, x), rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("shape", [1e30, 1e100, 1e308])
@pytest.mark.parametrize("x", [0.1, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 0.9])
def test_huge_symmetric_shapes(shape, x):
    np.testing.assert_allclose(
        bratio(shape, shape, x), tails(shape, shape, float(x)), rtol=5e-13, atol=5e-324
    )


@pytest.mark.parametrize("a,b,x", [(2.0, 3.0, 1e-160), (2.0, 3.0, 1e-162), (100.0, 100.0, 0.01)])
def test_small_tail_survives_rounded_unit_complement(a, b, x):
    p, q = bratio(a, b, x)
    assert p > 0 and q == 1
    np.testing.assert_allclose(p, tails(a, b, x)[0], rtol=5e-13, atol=5e-324)
    reverse = bratio(b, a, None, x)
    np.testing.assert_array_equal(reverse, (q, p))


@pytest.mark.parametrize(
    "args",
    [
        (np.inf, 1.0, 0.5),
        (1.0, np.nan, 0.5),
        (1.0, 1.0, np.nan),
        (1.0, 1.0, 0.5, np.inf),
        (1.0, 1.0, None),
    ],
)
def test_nonfinite_and_missing_inputs(args):
    with pytest.raises(ValueError):
        bratio(*args)


def test_coordinate_tolerance_boundary():
    eps = np.finfo(float).eps
    np.testing.assert_array_equal(bratio(1.0, 1.0, 0.5, 0.5 + 3 * eps), [0.5, 0.5])
    with pytest.raises(ValueError, match="IERR=5"):
        bratio(1.0, 1.0, 0.5, 0.5 + 4 * eps)


def test_broadcasting_and_independent_ownership():
    a = np.array([[0.0], [2.0], [1e308]])
    x = np.array([0.1, 0.5, 0.9])
    p, q = bratio(a, 3.0, x)
    target = [tails(float(v), 3.0, float(z)) for v in a[:, 0] for z in x]
    np.testing.assert_allclose(
        np.stack((p, q), axis=-1).reshape(-1, 2), target, rtol=5e-13, atol=5e-324
    )
    a[:] = 1
    x[:] = 0
    np.testing.assert_array_equal(p[0], 1)
    assert not np.shares_memory(p, q)
    for value in (p, q):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert bratio(np.empty((0, 1)), 3.0, np.full((1, 4), 0.5))[0].shape == (0, 4)


@pytest.mark.parametrize("a,b", [(2.5, 3.0), (50.0, 15.0)])
@pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
def test_finite_positive_identity_against_independent_integrals(a, b, x):
    np.testing.assert_allclose(
        integer_companion_tails(a, b, x, 1 - x), tails(a, b, x), rtol=2e-15, atol=0
    )


@pytest.mark.parametrize("a", [5e-324, 1e-20, 0.01, 0.5, 1.0, 100.0, 1e308])
@pytest.mark.parametrize("x", [5e-324, 0.1, 0.5, 0.9])
def test_unit_shape_against_exact_power_identity(a, x):
    with localcontext() as ctx:
        ctx.prec = 800
        aa, xx = Decimal.from_float(a), Decimal.from_float(x)
        logp = aa * xx.ln()
        p = Decimal(0) if logp < -800 else logp.exp()
        target = float(p), float(1 - p)
    np.testing.assert_allclose(bratio(a, 1.0, x), target, rtol=5e-13, atol=5e-324)
    np.testing.assert_allclose(bratio(1.0, a, None, x), target[::-1], rtol=5e-13, atol=5e-324)


def test_large_mixed_batch_preserves_independent_tails():
    shapes = np.array([0.0, 1e-20, 0.5, 2.0, 100.0])
    a = np.tile(shapes, 1000)
    p, q = bratio(a, 3.0, 0.001)
    target = np.tile(np.array([tails(float(v), 3.0, 0.001) for v in shapes]), (1000, 1))
    np.testing.assert_allclose(np.stack((p, q), axis=-1), target, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a,b", [(1000.0, 2000.0), (2000.0, 1000.0), (1000.0, 1000.0)])
@pytest.mark.parametrize("x", [0.01, 0.3, 0.49, 0.5, 0.51, 0.7, 0.99])
def test_large_shapes_against_positive_binomial_tails(a, b, x):
    np.testing.assert_allclose(bratio(a, b, x), tails(a, b, x), rtol=5e-13, atol=5e-324)
