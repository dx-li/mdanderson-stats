"""Independent chi-square identities, wide domains and unchanged C/F77 evidence."""

import json
from decimal import Decimal, localcontext
from math import erf, erfc, exp, expm1, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_chisq, cdfchi, cumchi

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_chisq.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]
TINY = np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("language,case", CASES)
def test_unchanged_native_references(language, case):
    w, p, q, x, df = case["input"]
    kw = dict(x=x, df=df)
    if w != 1:
        kw.update(p=p, q=q)
        name = {2: "x", 3: "df"}[w]
        truth = kw.pop(name)
    result = cdfchi(w, **kw)
    assert case["status"] == 0
    if w == 1:
        np.testing.assert_allclose([result.p, result.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        assert float(getattr(result, name)) == pytest.approx(truth, rel=3e-7, abs=1e-14)
        _, _, nx, nd, _ = case["result"]
        native = cdfchi(x=nx, df=nd)
        np.testing.assert_allclose([native.p, native.q], [p, q], rtol=3e-7, atol=1e-15)
    assert FIXTURE["profiles"][language]["adaptations"] == []


def integer_tails(x, df):
    with localcontext() as ctx:
        ctx.prec = 120
        z = Decimal.from_float(float(x)) / 2
        term = total = Decimal(1)
        for k in range(1, df // 2):
            term *= z / k
            total += term
        q = (-z).exp() * total
        return float(1 - q), float(q)


@pytest.mark.parametrize("df", [2, 4, 10, 20])
@pytest.mark.parametrize("x", [0.001, 0.1, 1, 10, 100])
def test_independent_decimal_finite_sum_all_modes(x, df):
    p, q = integer_tails(x, df)
    r = cdfchi(x=x, df=df)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdfchi(2, p=p, q=q, df=df).x, x, rtol=3e-11)
    np.testing.assert_allclose(cdfchi(3, p=p, q=q, x=x).df, df, rtol=3e-11)


@pytest.mark.parametrize("x", [TINY, 3 * TINY, 1e-308, 1e-200, 0.1, 1, 20, 100])
def test_independent_df_one_error_function(x):
    root = sqrt(x) / sqrt(2)
    p, q = erf(root), erfc(root)
    r = cdfchi(x=x, df=1)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=2e-13, atol=0)
    np.testing.assert_allclose(cdfchi(2, p=p, q=q, df=1).x, x, rtol=3e-12, atol=TINY)
    assert float(cdfchi(3, p=p, q=q, x=x).df) == pytest.approx(1, rel=3e-12)


def tiny_df_tails(x, df):
    # Independently sum the lower-gamma power series at exact df/2 and x/2,
    # without first rounding either division to binary64.
    with localcontext() as ctx:
        ctx.prec = 400
        a = Decimal.from_float(float(df)) / 2
        z = Decimal.from_float(float(x)) / 2
        euler = Decimal("0.57721566490153286060651209008240243104215933593992")
        term = total = Decimal(1)
        for n in range(1, 500):
            term *= z / (a + n)
            total += term
        assert term < Decimal("1e-400")
        # O(a**2) in logGamma is far below the 400-digit working precision.
        p = (a * z.ln() - z + euler * a).exp() * total
        return float(p), float(1 - p)


@pytest.mark.parametrize("df", [TINY, 3 * TINY, 1e-320, 1e-308, 4e-308])
@pytest.mark.parametrize("x", [TINY, 1e-308, 0.2, 2, 20])
def test_independent_400_digit_series_without_halving_underflow(x, df):
    p, q = tiny_df_tails(x, df)
    r = cdfchi(x=x, df=df)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=TINY)
    if q > 0:
        inv = cdfchi(2, p=p, q=q, df=df)
        check = cdfchi(x=inv.x, df=df)
        np.testing.assert_allclose(check.q, q, rtol=3e-12, atol=TINY)
        # Only claim parameter precision when input/output are well resolved.
        if df >= 1e-308 and x >= 1e-308:
            np.testing.assert_allclose(inv.x, x, rtol=3e-10, atol=0)


@pytest.mark.parametrize("x", [TINY, 2 * TINY, 3 * TINY, 1e-308, 1e-100, 1, 100])
def test_df_two_exponential_identity_including_rounding_tie(x):
    r = cdfchi(x=x, df=2)
    with localcontext() as ctx:
        ctx.prec = 800
        expected = float(1 - (-Decimal.from_float(float(x)) / 2).exp())
    np.testing.assert_allclose(r.p, expected, rtol=3e-15, atol=0)
    assert float(r.q) == pytest.approx(exp(-x / 2), rel=3e-15)


def test_native_failures_and_actual_search_bounds():
    for profile in FIXTURE["profiles"].values():
        assert [c["status"] for c in profile["invalid_cases"]] == [-1, -4, -5, -2, -3, 3]
        wide = profile["wide_cases"]
        assert wide[0]["status"] == 0 and wide[0]["result"][0] == 2
        assert wide[1]["status"] == 1 and wide[2]["status"] == 2
        assert wide[5]["result"][:2] == [0, 1]
        assert wide[7]["status"] == 0 and wide[7]["result"][2] == 0
    np.testing.assert_allclose(cdfchi(2, p=1e-100, df=1).x, pi / 2 * 1e-200, rtol=3e-13, atol=0)
    for large in (1e100, 1e200, 1e308):
        r = cdfchi(x=large, df=large)
        assert float(r.p) == 0.5 and float(r.q) == 0.5
    for w, kw in [(2, dict(df=1e200)), (3, dict(x=1e200))]:
        with pytest.raises(ValueError):
            cdfchi(w, p=0.5, **kw)
    df = np.array([1e-100, 1, 3, 1e90, 1e100])
    x = np.array([1e-200, 0.5, 2, 1e90, 1e100])
    r = cdfchi(x=x, df=df)
    np.testing.assert_allclose(cdfchi(3, p=r.p, q=r.q, x=x).df, df, rtol=3e-12, atol=0)
    assert float(cdfchi(2, p=0.5, df=1e100).x) == 1e100
    assert float(cdfchi(2, p=0, df=2).x) == 0


def test_broadcast_owned_empty_and_f95_overlap():
    x = np.array([[0.0], [1.0], [10.0]])
    df = np.array([0.2, 1, 10.0])
    r = cdfchi(x=x, df=df)
    f95 = cdf_chisq(x=x, df=df)
    np.testing.assert_allclose([r.p, r.q], [f95.cum, f95.ccum], rtol=3e-14, atol=0)
    x[:] = 8
    df[:] = 8
    assert r.x[0, 0] == 0 and r.df[0, 0] == 0.2
    for name in ["p", "q", "x", "df"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "x"), (3, "df")]:
        kw = dict(x=[], df=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfchi(w, **kw), name).size == 0
    np.testing.assert_allclose(cumchi(2, 2), [-expm1(-1), exp(-1)], rtol=3e-14)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (1.0, {}),
        (1, dict(x=1)),
        (1, dict(df=1)),
        (1, dict(x=-1, df=2)),
        (1, dict(x=1, df=0)),
        (1, dict(x=np.inf, df=2)),
        (1, dict(x=1, df=np.nan)),
        (1, dict(x=1, df=2, p=0.5)),
        (2, dict(q=0, df=2)),
        (2, dict(p=0.4, q=0.4, df=2)),
        (2, dict(p=0.5, q=0.5 + 4 * np.finfo(float).eps, df=2)),
        (2, dict(p=1e-300, df=1)),
        (2, dict(p=0.5, df=TINY)),
        (3, dict(p=0, x=1)),
        (3, dict(p=0.5, x=0)),
        (3, dict(p=0.5, x=1, df=1)),
    ],
)
def test_invalid_unidentified_unrepresentable(which, kwargs):
    with pytest.raises(ValueError):
        cdfchi(which, **kwargs)
