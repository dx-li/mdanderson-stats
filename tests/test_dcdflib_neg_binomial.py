"""Legacy negative-binomial native comparisons and independent wide-domain identities."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdfnbn

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_neg_binomial.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]
INTEGER_CASES = [
    (language, c)
    for language, c in CASES
    if c["input"][0] == 1
    and float(c["input"][3]).is_integer()
    and float(c["input"][4]).is_integer()
]


@pytest.mark.parametrize("language,case", CASES)
def test_ordinary_legacy_reference_against_python(language, case):
    w, p, q, f, s, pr, cpr = case["input"]
    kwargs = dict(f=f, s=s, pr=pr, cpr=cpr)
    if w != 1:
        kwargs.update(p=p, q=q)
        if w == 4:
            kwargs.pop("pr")
            kwargs.pop("cpr")
        else:
            kwargs.pop({2: "f", 3: "s"}[w])
    result = cdfnbn(w, **kwargs)
    assert case["status"] == 0
    assert FIXTURE["profiles"][language]["adaptations"] == []
    if w == 1:
        np.testing.assert_allclose([result.p, result.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        _, _, nf, ns, nx, ny, _ = case["result"]
        native_forward = cdfnbn(f=nf, s=ns, pr=nx, cpr=ny)
        np.testing.assert_allclose(
            [native_forward.p, native_forward.q], [p, q], rtol=3e-7, atol=1e-15
        )
        names = ["pr", "cpr"] if w == 4 else [{2: "f", 3: "s"}[w]]
        for name in names:
            truth = dict(f=f, s=s, pr=pr, cpr=cpr)[name]
            np.testing.assert_allclose(getattr(result, name), truth, rtol=3e-7, atol=1e-14)


@pytest.mark.parametrize("language,case", INTEGER_CASES)
def test_native_tails_against_independent_120_digit_pmf_sum(language, case):
    _, _, _, f, s, pr, cpr = case["input"]
    with localcontext() as ctx:
        ctx.prec = 120
        # Normalize from the smaller supplied coordinate, as for the Python pair.
        if pr <= cpr:
            x = Decimal.from_float(pr)
            y = 1 - x
        else:
            y = Decimal.from_float(cpr)
            x = 1 - y
        term = total = x ** int(s)
        for k in range(1, int(f) + 1):
            term *= (Decimal(int(s) + k - 1) / k) * y
            total += term
        expected = [float(total), float(1 - total)]
    r = cdfnbn(f=f, s=s, pr=pr, cpr=cpr)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize(
    "f,s,pr,cpr",
    [
        (0, 1e-308, 0.5, 0.5),
        (0, 1e-100, 0.5, 0.5),
        (0, 1e100, 1.0, 1e-100),
        (1e200, 1, 1e-200, 1.0),
        (1e100, 1, 1e-100, 1.0),
        (0, 1, 1e-300, 1.0),
        (1e-100, 1, 0.2, 0.8),
        (0, 1e-320, 0.5, 0.5),
    ],
)
def test_independent_800_digit_power_identities(f, s, pr, cpr):
    with localcontext() as ctx:
        ctx.prec = 800
        x = Decimal.from_float(pr) if pr <= cpr else 1 - Decimal.from_float(cpr)
        y = 1 - x
        a, b = Decimal.from_float(float(s)), Decimal.from_float(float(f)) + 1
        p = x**a if f == 0 else 1 - y**b
        pp, qq = float(p), float(1 - p)
    r = cdfnbn(f=f, s=s, pr=pr, cpr=cpr)
    tiny = np.nextafter(0.0, 1.0)
    np.testing.assert_allclose([r.p, r.q], [pp, qq], rtol=3e-13, atol=tiny)
    if qq > 0 and pp > 0:
        out = cdfnbn(4, p=pp, q=qq, f=f, s=s)
        check = cdfnbn(f=f, s=s, pr=out.pr, cpr=out.cpr)
        np.testing.assert_allclose([check.p, check.q], [pp, qq], rtol=3e-12, atol=2 * tiny)
        if 1e-300 <= s <= 1e100:
            np.testing.assert_allclose(
                cdfnbn(3, p=pp, q=qq, f=f, pr=pr, cpr=cpr).s, s, rtol=3e-7, atol=tiny
            )
        if 1e-10 <= f <= 1e100:
            np.testing.assert_allclose(
                cdfnbn(2, p=pp, q=qq, s=s, pr=pr, cpr=cpr).f, f, rtol=3e-7, atol=0
            )


def test_wide_bounds_native_repairs_and_endpoints():
    for v in (1e90, 1e100, 1e200, 1e308):
        assert float(cdfnbn(f=v, s=v, pr=0.5).p) == 0.5
    for mode, name in [(2, "f"), (3, "s")]:
        for v in (1e90, 1e100):
            kw = dict(f=v, s=v, pr=0.5, p=0.5)
            kw.pop(name)
            assert float(getattr(cdfnbn(mode, **kw), name)) == v
        kw = dict(f=1e200, s=1e200, pr=0.5, p=0.5)
        kw.pop(name)
        with pytest.raises((ValueError, ArithmeticError)):
            cdfnbn(mode, **kw)
    r = cdfnbn(4, p=1e-100, f=0, s=1)
    np.testing.assert_allclose(r.pr, 1e-100, rtol=3e-13, atol=0)
    r = cdfnbn(3, q=1e-100, f=0, pr=0.5)
    np.testing.assert_allclose(r.s, 1e-100 / np.log(2), rtol=3e-13, atol=0)
    assert float(cdfnbn(4, p=0, f=1, s=1).pr) == 0
    np.testing.assert_array_equal(cdfnbn(f=0, s=0, pr=[0, 0.5, 1]).p, 1)
    np.testing.assert_array_equal(cdfnbn(f=0, s=1, pr=[0, 1]).p, [0, 1])


def test_broadcast_owned_empty_and_mixed_endpoints():
    f = np.array([0.0, 0.5, 3, 1e90])
    s = np.array([1.0, 2, 4, 1e90])
    r = cdfnbn(f=f, s=s, pr=0.5)
    np.testing.assert_allclose(cdfnbn(2, p=r.p, q=r.q, s=s, pr=0.5).f, f, rtol=3e-12, atol=1e-14)
    f[:] = 8
    s[:] = 8
    assert r.f[0] == 0 and r.s[0] == 1
    for name in ["p", "q", "f", "s", "pr", "cpr"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "f"), (3, "s"), (4, "pr")]:
        kw = dict(f=[], s=[], pr=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfnbn(w, **kw), name).size == 0


@pytest.mark.parametrize(
    "which,kw",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(f=1)),
        (1, dict(f=-1, s=1, pr=0.5)),
        (1, dict(f=1, s=-1, pr=0.5)),
        (1, dict(f=1, s=1, pr=-0.1)),
        (1, dict(f=1, s=1, pr=0.4, cpr=0.4)),
        (1, dict(f=np.inf, s=1, pr=0.5)),
        (1, dict(f=1, s=np.nan, pr=0.5)),
        (2, dict(p=0.5, s=0, pr=0.5)),
        (2, dict(p=0.5, s=1, pr=0)),
        (2, dict(p=0, s=1, pr=0.5)),
        (2, dict(q=0, s=1, pr=0.5)),
        (2, dict(p=0.01, s=1, pr=0.5)),
        (3, dict(q=0, f=1, pr=0.5)),
        (4, dict(p=0.5, f=1, s=0)),
        (4, dict(q=0, f=1, s=1)),
        (4, dict(p=0.4, q=0.4, f=1, s=1)),
    ],
)
def test_invalid_and_unidentified(which, kw):
    with pytest.raises(ValueError):
        cdfnbn(which, **kw)


@pytest.mark.parametrize("large", [1e20, 1e200])
@pytest.mark.parametrize("small", [2, 3])
@pytest.mark.parametrize("reflect", [False, True])
def test_independent_disparate_shape_finite_sum(large, small, reflect):
    z = 1 / large
    f, s, pr, cpr = (small - 1, large, 1.0, z) if reflect else (large, small, z, 1.0)
    with localcontext() as ctx:
        ctx.prec = 800
        x = Decimal.from_float(z)
        b = Decimal.from_float(large) + (0 if reflect else 1)
        term = total = Decimal(1)
        for k in range(1, small):
            term *= (b + k - 1) * x / k
            total += term
        tail = ((1 - x).ln() * b).exp() * total
        p, q = (float(tail), float(1 - tail)) if reflect else (float(1 - tail), float(tail))
    r = cdfnbn(f=f, s=s, pr=pr, cpr=cpr)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-12, atol=0)
    out = cdfnbn(4, p=p, q=q, f=f, s=s)
    np.testing.assert_allclose([out.pr, out.cpr], [pr, cpr], rtol=3e-11, atol=0)
    w, name, truth = (2, "f", f) if reflect else (3, "s", s)
    kw = dict(f=f, s=s, pr=pr, cpr=cpr, p=p, q=q)
    kw.pop(name)
    np.testing.assert_allclose(getattr(cdfnbn(w, **kw), name), truth, rtol=3e-10, atol=1e-14)


@pytest.mark.parametrize("f", [1, 5, 20])
@pytest.mark.parametrize("s", [1e-308, 1e-320])
def test_independent_tiny_success_series_and_inverses(f, s):
    with localcontext() as ctx:
        ctx.prec = 400
        x = Decimal(".5")
        coefficient = -x.ln() - sum((1 - x) ** k / k for k in range(1, f + 1))
        q = float(Decimal.from_float(s) * coefficient)
    tiny = np.nextafter(0.0, 1.0)
    r = cdfnbn(f=f, s=s, pr=0.5)
    np.testing.assert_allclose(r.q, q, rtol=3e-13, atol=tiny)
    if q > 0:
        inv = cdfnbn(4, q=q, f=f, s=s)
        np.testing.assert_allclose(
            cdfnbn(f=f, s=s, pr=inv.pr, cpr=inv.cpr).q, q, rtol=3e-12, atol=tiny
        )
        inferred = cdfnbn(3, q=q, f=f, pr=0.5).s
        np.testing.assert_allclose(cdfnbn(f=f, s=inferred, pr=0.5).q, q, rtol=3e-12, atol=tiny)
        if s >= 1e-308:
            np.testing.assert_allclose(inferred, s, rtol=3e-8, atol=0)
            np.testing.assert_allclose(inv.pr, 0.5, rtol=3e-8, atol=0)
