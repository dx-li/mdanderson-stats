"""Legacy beta implementation against native references and independent identities."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdfbet, cumbet

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_beta.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]
INTEGER = [
    (lang, c)
    for lang, c in CASES
    if c["input"][0] == 1
    and float(c["input"][5]).is_integer()
    and float(c["input"][6]).is_integer()
]


@pytest.mark.parametrize("language,case", CASES)
def test_legacy_api_against_native_references(language, case):
    w, p, q, x, cx, a, b = case["input"]
    kw = dict(x=x, cx=cx, a=a, b=b)
    if w != 1:
        kw.update(p=p, q=q)
        for name in {2: ["x", "cx"], 3: ["a"], 4: ["b"]}[w]:
            kw.pop(name)
    r = cdfbet(w, **kw)
    assert FIXTURE["profiles"][language]["adaptations"] == []
    assert case["execution_outcome"] == "completed" and case["status"] == 0
    if w == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        for name in {2: ["x", "cx"], 3: ["a"], 4: ["b"]}[w]:
            truth = dict(x=x, cx=cx, a=a, b=b)[name]
            np.testing.assert_allclose(getattr(r, name), truth, rtol=3e-11, atol=0)
        _, _, nx, ny, na, nb, _ = case["result"]
        check = cdfbet(x=nx, cx=ny, a=na, b=nb)
        np.testing.assert_allclose([check.p, check.q], [p, q], rtol=1e-6, atol=0)


@pytest.mark.parametrize("language,case", INTEGER)
def test_integer_shapes_against_120_digit_binomial_sum(language, case):
    _, _, _, x, cx, a, b = case["input"]
    with localcontext() as ctx:
        ctx.prec = 120
        xx = Decimal.from_float(x) if x <= cx else 1 - Decimal.from_float(cx)
        yy = 1 - xx
        n = int(a + b - 1)
        term = yy**n
        lower = Decimal(0)
        upper = term
        for k in range(1, n + 1):
            term *= Decimal(n - k + 1) / k * xx / yy
            if k >= a:
                lower += term
            else:
                upper += term
        expected = [float(lower), float(upper)]
    r = cdfbet(x=x, cx=cx, a=a, b=b)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize("shape", [1e-308, 1e-320])
@pytest.mark.parametrize("x", [1e-300, 1e-100, 0.01, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("reflect", [False, True])
def test_tiny_shape_half_shape_independent_400_digit_integral(shape, x, reflect):
    # Q(a,1/2,x)/a -> 2*atanh(sqrt(1-x)); O(a) relative correction.
    with localcontext() as ctx:
        ctx.prec = 400
        y = 1 - Decimal.from_float(x)
        root = y.sqrt()
        expected = float(Decimal.from_float(shape) * ((1 + root) / (1 - root)).ln())
    r = cdfbet(cx=x, a=0.5, b=shape) if reflect else cdfbet(x=x, a=shape, b=0.5)
    np.testing.assert_allclose(
        r.p if reflect else r.q, expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0)
    )
    # At subnormal probabilities the coordinate has limited inverse precision.
    inv = cdfbet(2, p=r.p, q=r.q, a=r.a, b=r.b)
    check = cdfbet(x=inv.x, cx=inv.cx, a=r.a, b=r.b)
    np.testing.assert_allclose(
        [check.p, check.q], [r.p, r.q], rtol=3e-10, atol=32 * np.nextafter(0.0, 1.0)
    )
    if shape >= 1e-308:
        np.testing.assert_allclose(inv.cx if reflect else inv.x, x, rtol=3e-10)


@pytest.mark.parametrize(
    "a,b", [(1e-308, 2e-308), (1e-320, 2e-320), (1e-308, 1e-100), (1e-100, 1e-308)]
)
def test_two_tiny_shapes_independent_ratio_and_median(a, b):
    # First-order coordinate corrections are <= O(1e-100) here.
    with localcontext() as ctx:
        ctx.prec = 800
        aa, bb = map(Decimal.from_float, (a, b))
        expected = [float(bb / (aa + bb)), float(aa / (aa + bb))]
    r = cdfbet(x=0.2, a=a, b=b)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-13, atol=0)
    inv = cdfbet(2, p=r.p, q=r.q, a=a, b=b)
    np.testing.assert_allclose([inv.x, inv.cx], [0.5, 0.5], rtol=0, atol=0)


def _atan_decimal(x):
    term = total = x
    for n in range(1, 500):
        term *= -x * x
        total += term / (2 * n + 1)
    return total


@pytest.mark.parametrize("x", [1e-300, 1e-100, 0.01, 0.2, 0.5, 0.9])
def test_arcsine_independent_150_digit_identity(x):
    with localcontext() as ctx:
        ctx.prec = 150
        xx = Decimal.from_float(x)
        z = min(xx, 1 - xx)
        pi = 16 * _atan_decimal(Decimal(1) / 5) - 4 * _atan_decimal(Decimal(1) / 239)
        lower = 4 / pi * _atan_decimal(z.sqrt() / (1 + (1 - z).sqrt()))
        expected = (
            [float(lower), float(1 - lower)] if xx <= 0.5 else [float(1 - lower), float(lower)]
        )
    r = cdfbet(x=x, a=0.5, b=0.5)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-13, atol=0)
    inv = cdfbet(2, p=r.p, q=r.q, a=0.5, b=0.5)
    np.testing.assert_allclose(inv.x, x, rtol=3e-12, atol=0)


@pytest.mark.parametrize("n", [1e90, 1e100, 1e200, 1e308])
def test_huge_symmetry_and_distinct_shape_bounds(n):
    r = cdfbet(x=0.5, a=n, b=n)
    assert float(r.p) == 0.5
    np.testing.assert_allclose(cdfbet(2, p=0.5, a=n, b=n).x, 0.5, rtol=3e-14)
    for w in (3, 4):
        kw = dict(p=0.5, x=0.5)
        kw["b" if w == 3 else "a"] = n
        if n <= 1e100:
            assert float(getattr(cdfbet(w, **kw), "a" if w == 3 else "b")) == n
        else:
            with pytest.raises(ValueError):
                cdfbet(w, **kw)


def test_native_false_success_repairs_and_boundary_choices():
    for w in (3, 4):
        kw = dict(x=0.5)
        kw["q" if w == 3 else "p"] = 1e-100
        kw["b" if w == 3 else "a"] = 1
        r = cdfbet(w, **kw)
        np.testing.assert_allclose(
            getattr(r, "a" if w == 3 else "b"), 1e-100 / np.log(2), rtol=3e-13, atol=0
        )
    for target in (0.0, 1e-100):
        r = cdfbet(2, p=target, a=1, b=1)
        np.testing.assert_allclose(r.x, target, rtol=3e-13, atol=0)
        r = cdfbet(2, q=target, a=1, b=1)
        np.testing.assert_allclose(r.cx, target, rtol=3e-13, atol=0)
    np.testing.assert_array_equal(cdfbet(x=[0, 1], a=1e-308, b=1e308).p, [0, 1])


def test_broadcast_owned_arrays_mixed_and_empty():
    a = np.array([[0.5], [1.0]])
    r = cdfbet(x=[0.2, 0.8], a=a, b=2)
    a[:] = 4
    assert r.a[0, 0] == 0.5
    for name in ["p", "q", "x", "cx", "a", "b"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "x"), (3, "a"), (4, "b")]:
        kw = dict(x=[], a=[], b=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfbet(w, **kw), name).size == 0
    r = cdfbet(x=[0.2, 0.5, 0.8], a=[0.5, 1e90, 1], b=[0.25, 1e90, 1e-308])
    for w in (2, 3):
        kw = dict(p=r.p, q=r.q, x=r.x, cx=r.cx, a=r.a, b=r.b)
        for name in {2: ["x", "cx"], 3: ["a"], 4: ["b"]}[w]:
            kw.pop(name)
        inv = cdfbet(w, **kw)
        np.testing.assert_allclose(
            getattr(inv, "x" if w == 2 else "a"), r.x if w == 2 else r.a, rtol=3e-10
        )
    np.testing.assert_allclose(cumbet(None, 1, 1, cx=0.2), [0.8, 0.2], rtol=3e-14)


@pytest.mark.parametrize(
    "w,kw",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(x=0.5, b=1)),
        (1, dict(x=0.5, a=0, b=1)),
        (1, dict(x=0.5, a=1, b=-1)),
        (1, dict(x=0.5, a=np.inf, b=1)),
        (1, dict(x=-1, a=1, b=1)),
        (1, dict(x=0.4, cx=0.4, a=1, b=1)),
        (2, dict(p=0.4, q=0.4, a=1, b=1)),
        (2, dict(p=-1, a=1, b=1)),
        (3, dict(p=0, x=0.5, b=1)),
        (4, dict(q=0, x=0.5, a=1)),
        (3, dict(p=0, x=0, b=1)),
        (4, dict(p=1, x=1, a=1)),
        (2, dict(p=0.1, a=1e-308, b=1e-308)),
        (3, dict(q=1e-200, x=0.5, b=1)),
        (4, dict(p=1e-200, x=0.5, a=1)),
        (1, dict(p=0.5, x=0.5, a=1, b=1)),
    ],
)
def test_invalid_unidentified_and_unrepresentable(w, kw):
    with pytest.raises(ValueError):
        cdfbet(w, **kw)


@pytest.mark.parametrize(
    "a,b,x", [(1e-308, 1, 0.5), (1, 1e-308, 0.5), (1, 1e200, 1e-200), (1e200, 1, 1.0)]
)
def test_independent_800_digit_power_shapes(a, b, x):
    cx = 1e-200 if x == 1 else 1 - x
    with localcontext() as ctx:
        ctx.prec = 800
        xx = Decimal.from_float(x) if x <= cx else 1 - Decimal.from_float(cx)
        if b == 1:
            p = xx ** Decimal.from_float(a)
            q = 1 - p
        else:
            q = (1 - xx) ** Decimal.from_float(b)
            p = 1 - q
        expected = [float(p), float(q)]
    r = cdfbet(x=x, cx=cx, a=a, b=b)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0))
    inv = cdfbet(2, p=r.p, q=r.q, a=a, b=b)
    np.testing.assert_allclose([inv.x, inv.cx], [x, cx], rtol=3e-12, atol=0)


@pytest.mark.parametrize("w", [3, 4])
def test_shape_lower_upper_endpoints_and_tiny_fixed_shape(w):
    for target, fixed in [(1e-100, 1.0), (1e100, 1e100), (1e-50, 1e-308)]:
        a, b = (target, fixed) if w == 3 else (fixed, target)
        r = cdfbet(x=0.5, a=a, b=b)
        kw = dict(p=r.p, q=r.q, x=0.5)
        kw["b" if w == 3 else "a"] = fixed
        inv = cdfbet(w, **kw)
        np.testing.assert_allclose(getattr(inv, "a" if w == 3 else "b"), target, rtol=3e-12, atol=0)
