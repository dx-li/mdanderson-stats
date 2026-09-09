"""Legacy t native contracts, independent identities and wide domains."""

import json
from math import atan, erfc, exp, lgamma, log, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_t, cdft, cumt

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_t.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]


@pytest.mark.parametrize("language,case", CASES)
def test_native_c_and_fortran(language, case):
    w, p, q, t, df = case["input"]
    kw = dict(t=t, df=df)
    if w != 1:
        kw.update(p=p, q=q)
        name = "t" if w == 2 else "df"
        truth = kw.pop(name)
    r = cdft(w, **kw)
    if w == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        assert float(getattr(r, name)) == pytest.approx(truth, rel=3e-7, abs=1e-14)
        if case["status"] == 0:
            _, _, nt, nd, _ = case["result"]
            np.testing.assert_allclose(cumt(nt, nd), [p, q], rtol=3e-7, atol=1e-14)
    assert FIXTURE["profiles"][language]["adaptations"] == []


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_invalid_cases(language):
    assert [c["status"] for c in FIXTURE["profiles"][language]["invalid_cases"]] == [
        -1,
        -5,
        -2,
        -3,
        3,
    ]


@pytest.mark.parametrize("t", [1e-100, 0.1, 1, 10, 1e100, 1e154, 1e155, 1e160, 1e161, 1e200, 1e308])
def test_independent_cauchy_tails_and_quantiles(t):
    small = atan(1 / t) / pi
    for sign in [-1, 1]:
        r = cdft(t=sign * t, df=1)
        np.testing.assert_allclose(r.p if sign < 0 else r.q, small, rtol=3e-13, atol=0)
        if t <= 1e100 and r.p != r.q:
            assert float(cdft(2, p=r.p, q=r.q, df=1).t) == pytest.approx(sign * t, rel=3e-13)


@pytest.mark.parametrize("t", [0.1, 1, 10, 1e100, 1e150])
def test_independent_df_two_and_df_inversion(t):
    small = 1 / (sqrt(t * t + 2) * (sqrt(t * t + 2) + t))
    r = cdft(t=t, df=2)
    np.testing.assert_allclose(r.q, small, rtol=3e-13, atol=0)
    if t < 1e100:
        assert float(cdft(2, q=small, df=2).t) == pytest.approx(t, rel=3e-13)
    if small > 1e-250:
        assert float(cdft(3, q=small, t=t).df) == pytest.approx(2, rel=3e-12)


@pytest.mark.parametrize("df", [0.1, 0.2, 0.7, 1.5])
@pytest.mark.parametrize("t", [1e200, 1e308])
def test_logarithmic_tail_against_independent_gamma_identity(df, t):
    # x=df/(df+t*t) underflows; the first beta term has relative error O(x).
    # Compute the normalizer independently using standard-library log gamma.
    a = df / 2
    logtail = (
        a * (log(df) - 2 * log(t)) - log(a) - lgamma(a) - lgamma(0.5) + lgamma(a + 0.5) - log(2)
    )
    expected = exp(logtail)
    np.testing.assert_allclose(cumt(t, df)[1], expected, rtol=3e-13, atol=0)


@pytest.mark.parametrize("df", [1e-12, 1e-4, 0.2, 1, 20, 1e10])
def test_df_search_domains_and_mixed_endpoint_batch(df):
    r = cdft(t=[0.5, 2, 10], df=df)
    out = cdft(3, p=r.p, q=r.q, t=r.t)
    # Very small df is weakly identified by a probability rounded near 1/2.
    np.testing.assert_allclose(out.df, df, rtol=1e-4 if df == 1e-12 else 3e-8)


def test_wide_df_normal_limit_and_subnormal_df():
    for df in [1e20, 1e100, 1e308]:
        r = cdft(t=[-2.0, 0, 2.0], df=df)
        p = erfc(2 / sqrt(2)) / 2
        np.testing.assert_allclose(r.p, [p, 0.5, 1 - p], rtol=3e-13)
        np.testing.assert_allclose(cdft(2, p=r.p, q=r.q, df=df).t, r.t, rtol=3e-13)
    np.testing.assert_array_equal(
        cumt([0, 1, 1e308], np.nextafter(0.0, 1.0)), [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    )
    assert float(cdft(2, p=0.5, df=np.nextafter(0.0, 1.0)).t) == 0


def test_owned_broadcasts_empty_f95_overlap_and_batch_df():
    t = np.array([[-2.0], [0.5], [3.0]])
    df = np.array([0.01, 1.0, 10.0])
    r = cdft(t=t, df=df)
    f95 = cdf_t(t=t, df=df)
    np.testing.assert_allclose([r.p, r.q], [f95.cum, f95.ccum], rtol=3e-13)
    np.testing.assert_allclose(cdft(3, p=r.p, q=r.q, t=r.t).df, r.df, rtol=3e-11)
    t[:] = 8
    df[:] = 8
    assert r.t[0, 0] == -2 and r.df[0, 0] == 0.01
    for name in ["p", "q", "t", "df"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert cdft(t=[], df=1).p.size == 0
    assert cdft(2, p=[], df=1).t.size == 0
    assert cdft(3, p=[], t=[]).df.size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (1.0, {}),
        (1, dict(t=1)),
        (1, dict(df=1)),
        (1, dict(t=1, df=0)),
        (1, dict(t=np.inf, df=1)),
        (1, dict(t=1, df=np.nan)),
        (1, dict(t=1, df=1, p=0.5)),
        (2, dict(p=0, df=1)),
        (2, dict(q=0, df=1)),
        (2, dict(p=0.4, q=0.4, df=1)),
        (2, dict(p=0.5, q=0.5 + 4 * np.finfo(float).eps, df=1)),
        (2, dict(p=1e-200, df=1)),
        (2, dict(p=0.1, df=np.nextafter(0.0, 1.0))),
        (3, dict(p=0.5, t=0)),
        (3, dict(p=0.5, t=1)),
        (3, dict(p=0.1, t=1)),
        (3, dict(p=0.999999, t=0.1)),
    ],
)
def test_invalid_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdft(which, **kwargs)


@pytest.mark.parametrize("df", [np.nextafter(0.0, 1.0), 1e-323, 1e-320, 1e-308])
def test_tiny_df_does_not_lose_half_probability_to_beta_normalizer_overflow(df):
    # For finite t these df give deviations from 1/2 below probability spacing.
    np.testing.assert_array_equal(cumt([-1e308, 1, 1e100, 1e308], df), np.full((2, 4), 0.5))


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_wide_domain_tail_defects(language):
    cases = FIXTURE["profiles"][language]["wide_cases"]
    for c in cases[:2]:
        assert c["status"] == 0 and c["result"][:2] == [1.0, 0.0]
        _, _, _, t, df = c["input"]
        assert float(cumt(t, df)[1]) > 0
    for c in cases[2:]:
        _, _, _, t, df = c["input"]
        np.testing.assert_allclose(cumt(t, df), c["result"][:2], rtol=3e-13)


@pytest.mark.parametrize("df", [1e20, 1e100, 1e308])
@pytest.mark.parametrize("p", [1e-10, 1e-100, 1e-300])
def test_large_df_quantile_limit_against_independent_erfc(df, p):
    r = cdft(2, p=p, df=df)
    np.testing.assert_allclose(erfc(-float(r.t) / sqrt(2)) / 2, p, rtol=3e-12, atol=0)
