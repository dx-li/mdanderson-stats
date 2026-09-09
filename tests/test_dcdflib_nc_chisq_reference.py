"""Legacy noncentral chi-square contracts and independent failure evidence."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_nc_chisq, cdfchi

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
        kw["cum"] = p
    r = cdf_nc_chisq(w, **kw)
    if w == 1:
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=0, atol=2e-6)
    else:
        # Native solvers invert their approximate lower CDF. Validate the
        # Python answer against the actual requested probability separately.
        check = cdf_nc_chisq(x=r.x, df=r.df, pnonc=r.pnonc)
        np.testing.assert_allclose(check.cum, p, rtol=3e-12, atol=0)
        _, _, nx, ndf, nnc, _ = case["result"]
        check = cdf_nc_chisq(x=nx, df=ndf, pnonc=nnc)
        np.testing.assert_allclose(check.cum, p, rtol=0, atol=2e-6)


@pytest.mark.parametrize("language,case", INTEGER)
def test_independent_120_digit_poisson_mixtures(language, case):
    _, _, _, x, df, nc = case["input"]
    expected = poisson_mixture(x, int(df / 2), nc)
    np.testing.assert_allclose(case["result"][:2], expected, rtol=0, atol=2e-6)
    r = cdf_nc_chisq(x=x, df=df, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], expected, rtol=3e-13, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_ignored_q_invalid_bounds_and_boundary_ambiguities(language):
    d = FIXTURE["profiles"][language]
    assert [c["status"] for c in d["invalid_cases"]] == [-1, -1, -2, -2, -4, -5, -6]
    values = []
    for c in d["ignored_q_cases"]:
        assert c["status"] == 0
        assert c["result"][1] == c["input"][2]
        values.append(c["result"][2])
    np.testing.assert_array_equal(values, np.full(4, values[0]))
    assert d["boundary_cases"][7]["status"] == 0  # Executable accepts p=1-1e-16.
    for index, w in [(3, 3), (4, 4), (5, 3), (6, 4)]:
        c = d["boundary_cases"][index]
        assert c["status"] == 0
        kw = dict(cum=0, x=c["input"][3], df=c["input"][4], pnonc=c["input"][5])
        kw.pop("df" if w == 3 else "pnonc")
        with pytest.raises(ValueError):
            cdf_nc_chisq(w, **kw)
    assert d["boundary_cases"][2]["result"][2] == 0  # p=0 quantile is exactly zero.


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_independent_tiny_tail_and_inverse_false_successes(language):
    d = FIXTURE["profiles"][language]
    for index in (4, 8):
        c = d["wide_cases"][index]
        assert c["status"] == 0
        # The zero-th Poisson term alone is a rigorous lower bound.
        with localcontext() as ctx:
            ctx.prec = 800
            x = Decimal.from_float(c["input"][3])
            bound = (-Decimal(2)).exp() * (1 - (-x / 2).exp())
            assert bound > 0
            assert Decimal.from_float(c["result"][0]) < bound / 10
    c = d["wide_cases"][6]
    assert c["status"] == 0 and c["result"][2] == 0
    with localcontext() as ctx:
        ctx.prec = 150
        exact = float(-2 * (1 - Decimal.from_float(1e-100)).ln())
    np.testing.assert_allclose(cdfchi(2, p=1e-100, df=2).x, exact, rtol=3e-13, atol=0)
    c = d["wide_cases"][7]
    assert c["status"] == 0
    with localcontext() as ctx:
        ctx.prec = 150
        x = Decimal.from_float(c["result"][2])
        bound = (-Decimal(2)).exp() * (1 - (-x / 2).exp())
        assert bound > Decimal.from_float(c["input"][1]) * Decimal("1e40")
    for c in d["wide_cases"][:2]:
        assert c["status"] == 0 and c["result"][0] > 1
    c = d["wide_cases"][5]
    assert c["execution_outcome"] == "timeout" and c["timeout_seconds"] == 3
    assert c["status"] is None and c["result"] is None
    # The df->0 limit has a Poisson atom at zero plus central gamma terms.
    c = d["wide_cases"][3]
    assert c["status"] == 0
    np.testing.assert_allclose(c["result"][:2], poisson_mixture(1.0, 0, 4.0), rtol=0, atol=2e-6)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_ordinary_series_truncation_defeats_native_roundtrip(language):
    cases = FIXTURE["profiles"][language]["cases"]
    c = next(c for c in cases if c["input"] == [1, 0.0, 0.0, 0.2, 10.0, 20.0])
    expected, _ = poisson_mixture(0.2, 5, 20.0)
    native = c["result"][0]
    assert expected > native * 1e14
    inv = next(c for c in cases if c["input"][0] == 4 and c["input"][3:] == [0.2, 10.0, 20.0])
    assert inv["status"] == 0 and inv["result"][4] == 20
    # Solve the actual tiny target, rather than accepting a native roundtrip.
    modern = cdf_nc_chisq(4, cum=native, x=0.2, df=10)
    assert float(modern.pnonc) > 4 * 20
    np.testing.assert_allclose(
        cdf_nc_chisq(x=0.2, df=10, pnonc=modern.pnonc).cum, native, rtol=3e-12, atol=0
    )
