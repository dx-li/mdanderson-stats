"""Native t references and independent Cauchy/df=2 identities."""

import json
from math import atan2, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_t, cdf_t, cum_t, inv_t

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_t.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_t(case):
    which, p, q, t, df = case["input"]
    kwargs = dict(cum=p, ccum=q, t=t, df=df)
    for name in {1: ("cum", "ccum"), 2: ("t",), 3: ("df",)}[which]:
        kwargs.pop(name)
    r = cdf_t(which, **kwargs)
    if which == 1:
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        np.testing.assert_allclose([r.t, r.df], [t, df], rtol=3e-10, atol=1e-13)


@pytest.mark.parametrize("t", [-1e50, -5, -0.1, 0, 0.1, 5, 1e50])
@pytest.mark.parametrize("df", [1, 2])
def test_independent_cauchy_and_df_two(t, df):
    magnitude = abs(t)
    radius = sqrt(t * t + 2)
    small = atan2(1, magnitude) / pi if df == 1 else 1 / (radius * (radius + magnitude))
    p, q = (small, 1 - small) if t < 0 else (1 - small, small)
    r = cdf_t(t=t, df=df)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(inv_t(p, df, ccum=q)) == pytest.approx(t, rel=3e-12, abs=1e-14)
    if t:
        assert float(cdf_t(3, t=t, cum=p, ccum=q).df) == pytest.approx(df, rel=3e-11)


@pytest.mark.parametrize("df", [0.001, 0.1, 1, 30, 1e5, 1e10])
def test_df_magnitudes_and_endpoints(df):
    r = cdf_t(t=2, df=df)
    assert float(cdf_t(3, t=2, cum=r.cum, ccum=r.ccum).df) == pytest.approx(df, rel=3e-7)
    assert float(inv_t(r.cum, df, ccum=r.ccum)) == pytest.approx(2, rel=3e-12)


def test_broadcast_helpers_ownership_and_empty():
    t = np.array([[-2.0], [0.0], [3.0]])
    df = np.array([0.2, 1, 10])
    r = cdf_t(t=t, df=df)
    np.testing.assert_array_equal(cum_t(t, df), r.cum)
    np.testing.assert_array_equal(ccum_t(t, df), r.ccum)
    np.testing.assert_allclose(inv_t(r.cum, df, ccum=r.ccum), r.t, atol=1e-13)
    t[:] = 0
    df[:] = 1
    assert float(r.t[0, 0]) == -2 and float(r.df[0, 0]) == 0.2
    for name in ("cum", "ccum", "t", "df"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which in (1, 2, 3):
        kwargs = dict(cum=[], ccum=[], t=[], df=[])
        for name in {1: ("cum", "ccum"), 2: ("t",), 3: ("df",)}[which]:
            kwargs.pop(name)
        assert cdf_t(which, **kwargs).cum.size == 0
    assert float(inv_t(0.5, 0.001)) == 0
    assert float(inv_t(None, 2, ccum=1e-100)) == pytest.approx(sqrt(0.5e100), rel=1e-13)


@pytest.mark.parametrize("t", [-1e100, 1e100])
def test_coordinate_boundaries(t):
    r = cdf_t(t=t, df=1)
    assert float(inv_t(r.cum, 1, ccum=r.ccum)) == pytest.approx(t, rel=3e-14)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (1.0, {}),
        (1, dict(df=1)),
        (1, dict(t=1)),
        (1, dict(t=np.inf, df=1)),
        (1, dict(t=1e101, df=1)),
        (1, dict(t=1, df=0)),
        (1, dict(t=1, df=1e11)),
        (1, dict(t=1, df=1, cum=0.5)),
        (2, dict(df=1, cum=0)),
        (2, dict(df=1, cum=1)),
        (2, dict(df=0.001, ccum=1e-100)),
        (3, dict(t=0, cum=0.5)),
        (3, dict(t=1, cum=0.5)),
        (3, dict(t=-1, cum=0.8)),
        (3, dict(t=1, ccum=1e-100)),
        (3, dict(t=1, cum=0.8, df=1)),
    ],
)
def test_invalid_or_unidentified_requests(which, kwargs):
    with pytest.raises(ValueError):
        cdf_t(which, **kwargs)


def test_strict_coordinate_input_bounds():
    for t in (np.nextafter(-1e100, -np.inf), np.nextafter(1e100, np.inf)):
        with pytest.raises(ValueError):
            cdf_t(t=t, df=1)
