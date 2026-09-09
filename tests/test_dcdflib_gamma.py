"""Legacy gamma references, independent Decimal sums and scaled inverse repairs."""

import json
from decimal import Decimal, localcontext
from math import erf, erfc, exp, expm1, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_gamma, cdfgam, cumgam

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_gamma.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]


@pytest.mark.parametrize("language,case", CASES)
def test_native_c_and_fortran(language, case):
    w, p, q, x, a, r = case["input"]
    kw = dict(x=x, shape=a, rate=r)
    if w != 1:
        kw.update(p=p, q=q)
        name = {2: "x", 3: "shape", 4: "rate"}[w]
        truth = kw.pop(name)
    result = cdfgam(w, **kw)
    assert case["status"] == 0
    if w == 1:
        np.testing.assert_allclose([result.p, result.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        assert float(getattr(result, name)) == pytest.approx(truth, rel=3e-7, abs=1e-14)
        _, _, nx, na, nr, _ = case["result"]
        forward = cdfgam(x=nx, shape=na, rate=nr)
        np.testing.assert_allclose([forward.p, forward.q], [p, q], rtol=3e-7, atol=1e-15)
    assert FIXTURE["profiles"][language]["adaptations"] == []


def integer_tails(x, a, rate):
    with localcontext() as ctx:
        ctx.prec = 120
        z = Decimal(str(x)) * Decimal(str(rate))
        term = total = Decimal(1)
        for k in range(1, a):
            term *= z / k
            total += term
        q = (-z).exp() * total
        return float(1 - q), float(q)


@pytest.mark.parametrize("a", [1, 2, 5, 10])
@pytest.mark.parametrize("x", [0.01, 0.1, 1, 10])
@pytest.mark.parametrize("rate", [0.2, 3])
def test_independent_decimal_poisson_sum_all_inversions(a, x, rate):
    p, q = integer_tails(x, a, rate)
    r = cdfgam(x=x, shape=a, rate=rate)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    for w, name, truth in [(2, "x", x), (3, "shape", a), (4, "rate", rate)]:
        kw = dict(p=p, q=q, x=x, shape=a, rate=rate)
        kw.pop(name)
        np.testing.assert_allclose(getattr(cdfgam(w, **kw), name), truth, rtol=3e-11, atol=0)


@pytest.mark.parametrize(
    "x,rate", [(1e-154, 1e-154), (1e-160, 1e-160), (1e-200, 1e-200), (1e-300, 1e-100)]
)
def test_independent_half_shape_with_underflowed_unit_coordinate(x, rate):
    root = sqrt(x) * sqrt(rate)
    p, q = erf(root), erfc(root)
    r = cdfgam(x=x, shape=0.5, rate=rate)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    assert float(cdfgam(2, p=p, q=q, shape=0.5, rate=rate).x) == pytest.approx(x, rel=3e-12, abs=0)
    assert float(cdfgam(4, p=p, q=q, x=x, shape=0.5).rate) == pytest.approx(rate, rel=3e-12, abs=0)
    assert float(cdfgam(3, p=p, q=q, x=x, rate=rate).shape) == pytest.approx(0.5, rel=3e-12)


def small_shape_tails(a, z):
    # Independent lower-gamma power series, at 400-digit precision. The
    # log-Gamma Taylor remainder is negligible for these shapes <=1e-13.
    with localcontext() as ctx:
        ctx.prec = 400
        a, z = Decimal.from_float(float(a)), Decimal.from_float(float(z))
        euler = Decimal("0.57721566490153286060651209008240243104215933593992")
        c2 = Decimal("0.8224670334241132182362075833230125946094749506034")
        c3 = Decimal("0.40068563438653142846657938717048333025499543078017")
        loggamma = -euler * a + c2 * a * a - c3 * a * a * a
        term = total = Decimal(1)
        for n in range(1, 500):
            term *= z / (a + n)
            total += term
        assert term < Decimal("1e-400")
        p = (a * z.ln() - z - loggamma).exp() * total
        return float(p), float(1 - p)


@pytest.mark.parametrize("a", [1e-13, 1e-100, 1e-300, 1e-308, 1e-320])
@pytest.mark.parametrize("z", [0.1, 1, 10])
def test_independent_high_precision_small_shape_series(a, z):
    p, q = small_shape_tails(a, z)
    r = cdfgam(x=z, shape=a)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-11, atol=2 * np.nextafter(0.0, 1.0))
    if q > 0:
        out = cdfgam(2, p=p, q=q, shape=a)
        forward = cdfgam(x=out.x, shape=a)
        np.testing.assert_allclose(forward.q, q, rtol=3e-12, atol=np.nextafter(0.0, 1.0))
        if a >= 1e-308:
            assert float(out.x) == pytest.approx(z, rel=3e-9)


def test_wide_inverse_results_and_large_shape_anchor():
    p, q = -expm1(-1), exp(-1)
    assert float(cdfgam(2, p=p, q=q, shape=1, rate=1e-200).x) == pytest.approx(1e200, rel=3e-14)
    assert float(cdfgam(4, p=p, q=q, x=1e-200, shape=1).rate) == pytest.approx(1e200, rel=3e-14)
    assert float(cdfgam(2, p=1e-200, shape=0.5, rate=1e-308).x) == pytest.approx(
        (pi / 4) * 1e-92, rel=3e-13
    )
    a = np.array([1e-100, 0.5, 3, 1e90, 1e100])
    x = np.array([1e-200, 1e-200, 2, 1e90, 1e100])
    rate = np.array([1e-200, 1e-200, 1, 1, 1])
    r = cdfgam(x=x, shape=a, rate=rate)
    np.testing.assert_allclose(cdfgam(3, p=r.p, q=r.q, x=x, rate=rate).shape, a, rtol=3e-12, atol=0)
    assert float(cdfgam(x=1e308, shape=2, rate=1e308).p) == 1


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_invalid_and_wide_contract_evidence(language):
    profile = FIXTURE["profiles"][language]
    assert [c["status"] for c in profile["invalid_cases"]] == [-1, -4, -5, -6, -3, 3]
    cases = profile["wide_cases"]
    assert cases[1]["status"] == 0 and cases[1]["result"][2] == 1e200
    assert cases[2]["status"] == 0 and cases[2]["result"][4] == 1e200
    assert cases[3]["result"][:2] == [0.0, 1.0] and cases[4]["status"] == 10
    assert cases[5]["status"] == 0 and cases[5]["result"][4] == 0
    with pytest.raises(ValueError):
        cdfgam(4, p=0, x=1, shape=2)


def test_defaults_broadcasts_owned_empty_and_f95_overlap():
    x = np.array([[0.0], [1.0], [10.0]])
    a = np.array([0.2, 1, 10.0])
    r = cdfgam(x=x, shape=a, rate=2)
    f95 = cdf_gamma(x=x, shape=a, rate=2)
    np.testing.assert_array_equal([r.p, r.q], [f95.cum, f95.ccum])
    np.testing.assert_allclose(cdfgam(2, p=r.p, q=r.q, shape=a, rate=2).x, r.x, rtol=3e-13, atol=0)
    x[:] = 8
    a[:] = 8
    assert r.x[0, 0] == 0 and r.shape[0, 0] == 0.2
    for name in ["p", "q", "x", "shape", "rate"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "x"), (3, "shape"), (4, "rate")]:
        kw = dict(x=[], shape=[], rate=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfgam(w, **kw), name).size == 0
    np.testing.assert_allclose(cumgam(1, 1), [-expm1(-1), exp(-1)], rtol=3e-14)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(x=1)),
        (1, dict(shape=1)),
        (1, dict(x=-1, shape=1)),
        (1, dict(x=1, shape=0)),
        (1, dict(x=1, shape=1, rate=0)),
        (1, dict(x=np.inf, shape=1)),
        (1, dict(x=1, shape=1, rate=np.nan)),
        (1, dict(x=1, shape=1, p=0.5)),
        (2, dict(q=0, shape=1)),
        (2, dict(p=0.4, q=0.4, shape=1)),
        (2, dict(p=0.5, q=0.5 + 4 * np.finfo(float).eps, shape=1)),
        (2, dict(p=0.5, shape=1, rate=np.nextafter(0.0, 1.0))),
        (2, dict(p=1e-300, shape=0.5)),
        (3, dict(p=0, x=1, rate=1)),
        (3, dict(p=0.5, x=0, rate=1)),
        (4, dict(p=0.5, x=0, shape=1)),
        (4, dict(p=0, x=1, shape=1)),
        (4, dict(p=0.9, x=np.nextafter(0.0, 1.0), shape=1)),
    ],
)
def test_invalid_unidentified_or_unrepresentable(which, kwargs):
    with pytest.raises(ValueError):
        cdfgam(which, **kwargs)


def test_independent_subnormal_tail_with_normal_unit_coordinate():
    # P(2,z)=1-exp(-z)*(1+z); 800 digits retain the quadratic term.
    with localcontext() as ctx:
        ctx.prec = 800
        z = Decimal.from_float(1e-160)
        expected = float(1 - (-z).exp() * (1 + z))
    r = cdfgam(x=1e-160, shape=2)
    np.testing.assert_allclose(r.p, expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0))
