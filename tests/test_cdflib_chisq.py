"""Native chi-square references and independent exponential/normal identities."""

import json
from math import erf, erfc, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_chisq, cdf_chisq, cum_chisq, inv_chisq

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_chisq.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_reference(case):
    which, p, q, x, df = case["input"]
    kwargs = dict(cum=p, ccum=q, x=x, df=df)
    for name in {1: ("cum", "ccum"), 2: ("x",), 3: ("df",)}[which]:
        kwargs.pop(name)
    r = cdf_chisq(which, **kwargs)
    if which == 1:
        # Forward status is corrupted by finalizing an unused native zero finder.
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=2e-12, atol=0)
    else:
        np.testing.assert_allclose([r.x, r.df], [x, df], rtol=3e-11)


@pytest.mark.parametrize("x", [1e-100, 0.001, 0.2, 1, 5, 30, 1000])
@pytest.mark.parametrize("df", [1, 2])
def test_independent_identities(x, df):
    p, q = (erf(sqrt(x / 2)), erfc(sqrt(x / 2))) if df == 1 else (-np.expm1(-x / 2), np.exp(-x / 2))
    r = cdf_chisq(x=x, df=df)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-12, atol=0)
    assert float(inv_chisq(p, df, ccum=q)) == pytest.approx(x, rel=3e-12, abs=0)
    assert float(cdf_chisq(3, x=x, cum=p, ccum=q).df) == pytest.approx(df, rel=3e-12)


@pytest.mark.parametrize("df", [0.001, 0.01, 1, 10, 1e5, 1e10])
def test_degree_boundaries_and_magnitudes(df):
    r = cdf_chisq(x=df, df=df)
    assert float(cdf_chisq(3, x=df, cum=r.cum, ccum=r.ccum).df) == pytest.approx(df, rel=1e-13)
    assert float(inv_chisq(r.cum, df, ccum=r.ccum)) == pytest.approx(df, rel=1e-13)


def test_broadcast_ownership_helpers_and_endpoints():
    x = np.array([[0.1], [1], [5]])
    df = np.array([1, 2, 3])
    r = cdf_chisq(x=x, df=df)
    np.testing.assert_array_equal(cum_chisq(x, df), r.cum)
    np.testing.assert_array_equal(ccum_chisq(x, df), r.ccum)
    np.testing.assert_allclose(inv_chisq(None, df, ccum=r.ccum), r.x)
    assert r.cum.shape == (3, 3)
    x[:] = 0
    df[:] = 0
    assert float(r.x[0, 0]) == 0.1 and float(r.df[0, 0]) == 1
    for name in ("x", "df", "cum", "ccum"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert float(cum_chisq(0, 1)) == 0
    assert float(ccum_chisq(0, 1)) == 1
    assert float(inv_chisq(0, 1)) == 0
    assert float(inv_chisq(None, 2, ccum=1e-300)) == pytest.approx(-2 * np.log(1e-300))
    assert cdf_chisq(x=[], df=1).cum.size == 0
    assert cdf_chisq(2, cum=[], df=1).x.size == 0
    assert cdf_chisq(3, cum=[], x=1).df.size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (1.0, {}),
        (0, {}),
        (4, {}),
        (1, {"x": 1}),
        (1, {"df": 1}),
        (1, {"x": 1, "df": 0}),
        (1, {"x": 1, "df": 1e11}),
        (1, {"x": 1, "df": np.nan}),
        (1, {"x": -1, "df": 1}),
        (1, {"x": 1e101, "df": 1}),
        (1, {"x": 1, "df": 1, "cum": 0.5}),
        (2, {"df": 1, "cum": 1}),
        (2, {"df": 1, "cum": 0.5, "x": 1}),
        (3, {"x": 1, "cum": 0.5, "df": 1}),
        (3, {"x": 0, "cum": 0.5}),
        (3, {"x": 1, "cum": 0}),
        (3, {"x": 1e20, "cum": 0.5}),
    ],
)
def test_invalid_requests(which, kwargs):
    with pytest.raises(ValueError):
        cdf_chisq(which, **kwargs)


def test_strict_df_input_bounds():
    for df in (np.nextafter(0.001, 0), np.nextafter(1e10, np.inf)):
        with pytest.raises(ValueError):
            cdf_chisq(x=1, df=df)
