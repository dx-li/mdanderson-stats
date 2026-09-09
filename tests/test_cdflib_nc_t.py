"""Closed forms, independent normal/chi-square integration and native evidence."""

import json
from math import erfc, exp, factorial, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_nc_t, cdf_nc_t, cdf_t, cum_nc_t, inv_nc_t

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_nc_t.json").read_text())


def phi(x):
    return erfc(-x / sqrt(2)) / 2


def df_two(t, nc):
    ratio = abs(t) / sqrt(t * t + 2)
    term = ratio * exp(-nc * nc / (t * t + 2))
    p = phi(-nc) + (term * phi(nc * ratio) if t >= 0 else -term * phi(-nc * ratio))
    q = phi(nc) - (term * phi(nc * ratio) if t >= 0 else -term * phi(-nc * ratio))
    return p, q


@pytest.mark.parametrize("t", [-5, -2, -0.5, 0, 0.5, 2, 5])
@pytest.mark.parametrize("nc", [0, 0.5, 2, 4])
def test_independent_df_two_closed_form(t, nc):
    p, q = df_two(t, nc)
    r = cdf_nc_t(t=t, df=2, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-10, atol=0)
    assert float(inv_nc_t(p, 2, nc, ccum=q)) == pytest.approx(t, rel=3e-9, abs=1e-10)
    assert float(cdf_nc_t(4, t=t, df=2, cum=p, ccum=q).pnonc) == pytest.approx(
        nc, rel=3e-9, abs=1e-11
    )
    if t != 0:
        assert float(
            cdf_nc_t(3, t=t, pnonc=nc, cum=p, ccum=q, df_bracket=(1, 3)).df
        ) == pytest.approx(2, rel=3e-8)


def independent_integral(t, df, nc, n):
    # E[Phi(t*sqrt(V/df)-nc)], integrating the chi-square radial density.
    # Integer even df gives its normalization using factorial alone.
    shape = df // 2
    r = np.linspace(0, 8, n + 1)
    density = 2 * shape**shape / factorial(shape - 1) * r ** (df - 1) * np.exp(-shape * r * r)
    p = np.array([phi(t * v - nc) for v in r]) * density
    q = np.array([phi(nc - t * v) for v in r]) * density
    return np.array(
        [(a[0] + a[-1] + 4 * a[1:-1:2].sum() + 2 * a[2:-1:2].sum()) * (8 / n) / 3 for a in (p, q)]
    )


@pytest.mark.parametrize("t,df,nc", [(0.5, 10, 4), (-2, 10, 4), (1, 10, 3), (3, 4, 0.5)])
def test_independent_chi_square_integral(t, df, nc):
    first = independent_integral(t, df, nc, 8192)
    second = independent_integral(t, df, nc, 16384)
    np.testing.assert_allclose(first, second, rtol=1e-11, atol=0)
    r = cdf_nc_t(t=t, df=df, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], second, rtol=3e-10, atol=0)


@pytest.mark.parametrize("case", FIXTURE["profiles"]["central_status_repaired"]["cases"])
def test_native_reference(case):
    which, p, q, t, df, nc = case["input"]
    bad = t == 0.5 and df == 10 and nc == 4
    if which == 1:
        r = cdf_nc_t(t=t, df=df, pnonc=nc)
        if bad:
            assert case["result"][0] == pytest.approx(phi(-nc), rel=1e-14)
            assert float(r.cum) > 7 * case["result"][0]
        else:
            np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=0, atol=1e-8)
    else:
        kwargs = dict(cum=p, ccum=q, t=t, df=df, pnonc=nc)
        name = {2: "t", 3: "df", 4: "pnonc"}[which]
        truth = kwargs.pop(name)
        if which == 3:
            kwargs["df_bracket"] = (df * 0.99, df * 1.01)
        if bad and which == 3:
            with pytest.raises(ValueError, match="not bracketed"):
                cdf_nc_t(which, **kwargs)
            return
        r = cdf_nc_t(which, **kwargs)
        if bad:
            assert abs(float(getattr(r, name)) - truth) > 1e-2
        else:
            assert float(getattr(r, name)) == pytest.approx(truth, rel=3e-3, abs=1e-5)
        forward = cdf_nc_t(t=r.t, df=r.df, pnonc=r.pnonc)
        assert float(forward.cum if p <= q else forward.ccum) == pytest.approx(min(p, q), rel=1e-8)


def test_native_profiles_and_multiple_root_selection():
    original = FIXTURE["profiles"]["original"]
    assert original["patches"] == []
    assert any(c["status"] == 10 for c in original["cases"])
    repaired = FIXTURE["profiles"]["central_status_repaired"]
    assert repaired["patches"][0]["file"] == "cdf_t_mod.f90"
    with pytest.raises(ValueError, match="df_bracket"):
        cdf_nc_t(3, t=1, pnonc=3, cum=0.03)
    r = cdf_nc_t(3, t=1, pnonc=3, cum=0.03, df_bracket=([0.001, 0.2], [0.2, 100]))
    assert float(r.df[0]) < 0.2 < float(r.df[1])
    np.testing.assert_allclose(cum_nc_t(1, r.df, 3), 0.03, rtol=1e-12)
    # Genuine sign changes independently surround both returned roots.
    for root in r.df:
        values = cum_nc_t(1, root * np.array([0.99, 1.01]), 3) - 0.03
        assert values[0] * values[1] < 0
    assert repaired["multiple_root_case"]["status"] != 0


@pytest.mark.parametrize("df", [0.001, 2, 1e10])
@pytest.mark.parametrize("nc", [0, 4, 37])
def test_zero_t_normal_identity(df, nc):
    p, q = phi(-nc), phi(nc)
    r = cdf_nc_t(t=0, df=df, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(cdf_nc_t(4, t=0, df=df, cum=p, ccum=q).pnonc) == pytest.approx(
        nc, rel=3e-13, abs=0
    )
    with pytest.raises(ValueError, match="unidentified"):
        cdf_nc_t(3, t=0, pnonc=nc, cum=p, ccum=q)


@pytest.mark.parametrize("t,df,nc", [(-2, 0.001, 0), (1, 1e10, 0), (1e4, 2, 1e4), (1e4, 1e10, 1e4)])
def test_bounds_and_central_reduction(t, df, nc):
    r = cdf_nc_t(t=t, df=df, pnonc=nc)
    assert float(inv_nc_t(r.cum, df, nc, ccum=r.ccum)) == pytest.approx(t, rel=5e-8)
    assert float(cdf_nc_t(4, t=t, df=df, cum=r.cum, ccum=r.ccum).pnonc) == pytest.approx(
        nc, rel=3e-10, abs=0
    )
    if nc == 0:
        central = cdf_t(t=t, df=df)
        np.testing.assert_array_equal([r.cum, r.ccum], [central.cum, central.ccum])
    assert float(cdf_nc_t(3, t=t, pnonc=nc, cum=r.cum, ccum=r.ccum).df) == pytest.approx(
        df, rel=3e-6
    )


def test_mixed_broadcasts_and_ownership():
    t = np.array([[-2.0], [0.0], [2.0]])
    nc = np.array([0.0, 0.5, 2.0])
    r = cdf_nc_t(t=t, df=2, pnonc=nc)
    np.testing.assert_array_equal(cum_nc_t(t, 2, nc), r.cum)
    np.testing.assert_array_equal(ccum_nc_t(t, 2, nc), r.ccum)
    np.testing.assert_allclose(inv_nc_t(r.cum, 2, nc, ccum=r.ccum), r.t, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        cdf_nc_t(4, t=t, df=2, cum=r.cum, ccum=r.ccum).pnonc, r.pnonc, rtol=1e-12, atol=0
    )
    t[:] = 3
    nc[:] = 3
    assert float(r.t[0, 0]) == -2 and float(r.pnonc[0, 0]) == 0
    for name in ("cum", "ccum", "t", "df", "pnonc"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert cdf_nc_t(t=[], df=2, pnonc=4).cum.size == 0
    assert inv_nc_t([], 2, 4).size == 0
    assert cdf_nc_t(3, t=[], pnonc=1, cum=[]).df.size == 0
    assert cdf_nc_t(4, t=[], df=1, cum=[]).pnonc.size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(t=1, df=1)),
        (1, dict(t=1, pnonc=1)),
        (1, dict(df=1, pnonc=1)),
        (1, dict(t=1e101, df=1, pnonc=1)),
        (1, dict(t=-1e101, df=1, pnonc=1)),
        (1, dict(t=1, df=0, pnonc=1)),
        (1, dict(t=1, df=1e11, pnonc=1)),
        (1, dict(t=1, df=1, pnonc=-1)),
        (1, dict(t=1, df=1, pnonc=1e5)),
        (1, dict(t=1, df=1, pnonc=1, cum=0.5)),
        (2, dict(t=1, df=1, pnonc=1, cum=0.5)),
        (3, dict(t=1, df=1, pnonc=1, cum=0.5)),
        (4, dict(t=1, df=1, pnonc=1, cum=0.5)),
        (2, dict(df=1, pnonc=1, cum=0)),
        (2, dict(df=1, pnonc=1, cum=1)),
        (3, dict(t=1, pnonc=1, cum=0.5, df_bracket=(2, 1))),
        (3, dict(t=1, pnonc=1, cum=0.5, df_bracket=(0, 2))),
        (1, dict(t=1, df=1, pnonc=1, df_bracket=(1, 2))),
        (4, dict(t=0, df=1, cum=0.9)),
        (3, dict(t=0, pnonc=1, cum=0.5)),
    ],
)
def test_invalid_unidentified_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdf_nc_t(which, **kwargs)


def test_negative_tail_kernel_failure_has_independent_repair():
    t, df, nc = -5.0, 2, 9.765625
    first = independent_integral(t, df, nc, 32768)
    second = independent_integral(t, df, nc, 65536)
    np.testing.assert_allclose(first, second, rtol=2e-8, atol=0)
    r = cdf_nc_t(t=t, df=df, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], second, rtol=2e-8, atol=0)
    assert float(cum_nc_t(-2, 2, 39.0625)) == 0


def test_large_noncentrality_against_df_two_identity():
    p, q = df_two(1e4, 1e4)
    r = cdf_nc_t(t=1e4, df=2, pnonc=1e4)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=1e-7, atol=0)


@pytest.mark.parametrize("bounds", [(1, 2, 3), (1e-4, 2), (1, 1e11), (2, 2)])
def test_invalid_df_brackets(bounds):
    with pytest.raises(ValueError):
        cdf_nc_t(3, t=1, pnonc=1, cum=0.3, df_bracket=bounds)
