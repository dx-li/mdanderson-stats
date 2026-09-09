"""Legacy noncentral chi-square implementation and independent identities."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdfchn, cumchn

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_nc_chisq.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]
INTEGER = [(lang, c) for lang, c in CASES if c["input"][0] == 1 and c["input"][4] in (2.0, 10.0)]


def poisson_mixture(x, shape, nc):
    with localcontext() as ctx:
        ctx.prec = 120
        z, mean = Decimal.from_float(x) / 2, Decimal.from_float(nc) / 2
        weight, exponential = (-mean).exp(), (-z).exp()
        p = q = Decimal(0)
        for j in range(250):
            if shape + j == 0:
                upper = Decimal(0)
            else:
                term = total = Decimal(1)
                for k in range(1, shape + j):
                    term *= z / k
                    total += term
                upper = exponential * total
            p += weight * (1 - upper)
            q += weight * upper
            weight *= mean / (j + 1)
        assert weight < Decimal("1e-100")
        return float(p), float(q)


@pytest.mark.parametrize("language,case", CASES)
def test_ordinary_native_cases_against_existing_python(language, case):
    w, p, q, x, df, nc = case["input"]
    assert case["status"] == 0 and case["execution_outcome"] == "completed"
    assert not FIXTURE["profiles"][language]["adaptations"]
    kw = dict(x=x, df=df, pnonc=nc)
    if w != 1:
        kw.pop({2: "x", 3: "df", 4: "pnonc"}[w])
        kw["p"] = p
    r = cdfchn(w, **kw)
    if w == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=0, atol=2e-6)
    else:
        # Native solvers invert their approximate lower CDF. Validate the
        # Python answer against the actual requested probability separately.
        check = cdfchn(x=r.x, df=r.df, pnonc=r.pnonc)
        np.testing.assert_allclose(check.p, p, rtol=3e-12, atol=0)
        _, _, nx, ndf, nnc, _ = case["result"]
        check = cdfchn(x=nx, df=ndf, pnonc=nnc)
        np.testing.assert_allclose(check.p, p, rtol=0, atol=2e-6)


@pytest.mark.parametrize("language,case", INTEGER)
def test_independent_120_digit_poisson_mixtures(language, case):
    _, _, _, x, df, nc = case["input"]
    expected = poisson_mixture(x, int(df / 2), nc)
    np.testing.assert_allclose(case["result"][:2], expected, rtol=0, atol=2e-6)
    r = cdfchn(x=x, df=df, pnonc=nc)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-13, atol=0)


@pytest.mark.parametrize("x", [1e-100, 1e-300, 1e-320])
@pytest.mark.parametrize("nc", [0.5, 4.0, 20.0])
def test_independent_800_digit_small_coordinate(x, nc):
    # For df=2 the positive mixture's leading term is exp(-nc/2)*x/2;
    # the relative correction is O(x*(1+nc)), negligible at these scales.
    with localcontext() as ctx:
        ctx.prec = 800
        expected = float((-Decimal.from_float(nc) / 2).exp() * Decimal.from_float(x) / 2)
    r = cdfchn(x=x, df=2, pnonc=nc)
    np.testing.assert_allclose(r.p, expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0))
    if x >= 1e-300:
        np.testing.assert_allclose(cdfchn(2, p=expected, df=2, pnonc=nc).x, x, rtol=3e-11, atol=0)


@pytest.mark.parametrize("df", [1e-100, 1e-308, np.nextafter(0.0, 1.0)])
def test_tiny_degrees_positive_noncentrality(df):
    expected = poisson_mixture(1.0, 0, 4.0)
    r = cdfchn(x=1, df=df, pnonc=4)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdfchn(2, p=r.p, df=df, pnonc=4).x, 1.0, rtol=3e-11)
    # At every representable tiny positive x, the vanishing-df component
    # tends to its Poisson mass, while x=0 itself remains an exact endpoint.
    np.testing.assert_allclose(
        cdfchn(x=np.nextafter(0.0, 1.0), df=df, pnonc=4).p, np.exp(-2), rtol=3e-13
    )
    assert float(cdfchn(x=0, df=df, pnonc=4).p) == 0


@pytest.mark.parametrize("nc", [0.5, 4.0, 100.0])
@pytest.mark.parametrize("x", [1e-12, 0.1, 1.0, 10.0])
def test_df_one_shifted_normal_identity(x, nc):
    from math import erfc, exp, pi, sqrt

    root, shift = sqrt(x), sqrt(nc)
    if root * (shift + 1) < 1e-4:
        p = 2 * root * exp(-nc / 2) / sqrt(2 * pi) * (1 + (nc - 1) * x / 6)
        q = 1 - p
    else:
        p = (erfc((shift - root) / sqrt(2)) - erfc((shift + root) / sqrt(2))) / 2
        q = (erfc((root - shift) / sqrt(2)) + erfc((root + shift) / sqrt(2))) / 2
    r = cdfchn(x=x, df=1, pnonc=nc)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-12, atol=0)


@pytest.mark.parametrize("size", [1e90, 1e100, 1e200, 1e308])
def test_wide_central_and_shifted_normal_medians(size):
    assert float(cdfchn(x=size, df=size, pnonc=0).p) == 0.5
    assert float(cdfchn(x=size, df=1, pnonc=size).p) == 0.5
    assert float(cdfchn(x=1, df=2, pnonc=size).p) == 0
    if size <= 1e100:
        np.testing.assert_allclose(cdfchn(2, p=0.5, df=1, pnonc=size).x, size, rtol=3e-14)
    else:
        with pytest.raises(ValueError):
            cdfchn(2, p=0.5, df=1, pnonc=size)


def test_native_false_success_repairs_and_ignored_q():
    expected = poisson_mixture(0.2, 5, 20.0)[0]
    r = cdfchn(x=0.2, df=10, pnonc=20)
    np.testing.assert_allclose(r.p, expected, rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdfchn(4, p=expected, x=0.2, df=10).pnonc, 20, rtol=3e-12)
    np.testing.assert_allclose(cdfchn(2, p=1e-100, df=2, pnonc=0).x, 2e-100, rtol=3e-13, atol=0)
    for q in (None, -123, np.nan, [[1, 2, 3], [4, 5, 6]], "ignored"):
        r = cdfchn(2, p=0.5, q=q, df=2, pnonc=4)
        assert r.x.shape == () and float(r.q) == 0.5
    assert float(cdfchn(2, p=0, df=2, pnonc=4).x) == 0
    # The source's executable upper p endpoint is accepted.
    r = cdfchn(2, p=1 - 1e-16, df=2, pnonc=0)
    assert np.isfinite(r.x)


def test_owned_broadcast_empty_and_parameter_endpoints():
    x = np.array([[0.2], [2.0]])
    r = cdfchn(x=x, df=[2, 10], pnonc=4)
    x[:] = 8
    assert r.x[0, 0] == 0.2
    for name in ["p", "q", "x", "df", "pnonc"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "x"), (3, "df"), (4, "pnonc")]:
        kw = dict(x=[], df=[], pnonc=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfchn(w, **kw), name).size == 0
    r = cdfchn(x=[1, 2, 1e90], df=[2, 2, 1e90], pnonc=[0, 4, 0])
    np.testing.assert_allclose(cdfchn(3, p=r.p, x=r.x, pnonc=r.pnonc).df, r.df, rtol=3e-12)
    r = cdfchn(x=[1, 10002], df=2, pnonc=[0, 1e4])
    np.testing.assert_allclose(cdfchn(4, p=r.p, x=r.x, df=2).pnonc, [0, 1e4], rtol=3e-12, atol=0)
    np.testing.assert_allclose(
        cumchn(1, 2, 4), [cdfchn(x=1, df=2, pnonc=4).p, cdfchn(x=1, df=2, pnonc=4).q]
    )


@pytest.mark.parametrize(
    "w,kw",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(x=1, df=2)),
        (1, dict(x=-1, df=2, pnonc=4)),
        (1, dict(x=1, df=0, pnonc=4)),
        (1, dict(x=1, df=2, pnonc=-1)),
        (1, dict(x=np.inf, df=2, pnonc=4)),
        (2, dict(q=0.5, df=2, pnonc=4)),
        (2, dict(p=1, df=2, pnonc=4)),
        (2, dict(p=-1, df=2, pnonc=4)),
        (3, dict(p=0, x=0, pnonc=4)),
        (4, dict(p=0, x=0, df=2)),
        (3, dict(p=0, x=1, pnonc=4)),
        (4, dict(p=0, x=1, df=2)),
        (3, dict(p=0.5, x=1e200, pnonc=0)),
        (4, dict(p=0.9, x=0.2, df=10)),
        (1, dict(p=0.5, x=1, df=2, pnonc=4)),
    ],
)
def test_invalid_unidentified_and_outside_bounds(w, kw):
    with pytest.raises(ValueError):
        cdfchn(w, **kw)


@pytest.mark.parametrize("df,nc", [(1e-308, 1e-100), (1e-308, 1e-308), (1e-320, 1e-320)])
def test_independent_small_noncentrality_retains_central_upper_tail(df, nc):
    with localcontext() as ctx:
        ctx.prec = 400
        z = Decimal(".5")
        euler = Decimal(
            "0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495"
        )
        term = -z
        series = term
        for k in range(2, 300):
            term *= -z / k
            series += term / k
        e1 = -euler - z.ln() - series
        expected = float(Decimal.from_float(df) * e1 / 2 + Decimal.from_float(nc) * (-z).exp() / 2)
    np.testing.assert_allclose(
        cdfchn(x=1, df=df, pnonc=nc).q, expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0)
    )
