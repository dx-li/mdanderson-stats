"""Native F references, Cauchy-square identities and reciprocal symmetry."""

import json
from math import atan, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_f, cdf_f, cum_f, inv_f

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_f.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_f(case):
    which, p, q, f, dfn, dfd = case["input"]
    r = cdf_f(f=f, dfn=dfn, dfd=dfd) if which == 1 else cdf_f(2, cum=p, ccum=q, dfn=dfn, dfd=dfd)
    if which == 1:
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        assert float(r.f) == pytest.approx(f, rel=3e-11)


@pytest.mark.parametrize("f", [0, 1e-100, 1e-20, 0.1, 1, 10, 1e100])
@pytest.mark.parametrize("df", [1, 2])
def test_independent_cauchy_square_and_uniform_beta(f, df):
    if df == 1:
        small = 2 * atan(sqrt(f) if f <= 1 else 1 / sqrt(f)) / pi
        p, q = (small, 1 - small) if f <= 1 else (1 - small, small)
    else:
        p, q = f / (1 + f), 1 / (1 + f)
    r = cdf_f(f=f, dfn=df, dfd=df)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(inv_f(p, df, df, ccum=q)) == pytest.approx(f, rel=3e-13, abs=0)


@pytest.mark.parametrize("dfn,dfd", [(0.001, 2), (2, 0.001), (1, 30), (30, 1), (1e10, 1e10)])
def test_reciprocal_symmetry_and_df_bounds(dfn, dfd):
    f = np.array([0.1, 1, 10.0])
    r = cdf_f(f=f, dfn=dfn, dfd=dfd)
    other = cdf_f(f=1 / f, dfn=dfd, dfd=dfn)
    np.testing.assert_allclose(r.cum, other.ccum, rtol=3e-13, atol=0)
    if dfn == dfd == 1e10:
        np.testing.assert_array_equal(r.cum, [0, 0.5, 1])
        with pytest.raises(ValueError):
            inv_f(r.cum, dfn, dfd, ccum=r.ccum)
    representable = (r.cum > 0) & (r.ccum > 0)
    np.testing.assert_allclose(
        inv_f(r.cum[representable], dfn, dfd, ccum=r.ccum[representable]),
        f[representable],
        rtol=3e-12,
    )


@pytest.mark.parametrize("df", [0.001, 1, 1e5, 1e10])
def test_equal_df_midpoint(df):
    assert float(cum_f(1, df, df)) == 0.5
    assert float(inv_f(0.5, df, df)) == pytest.approx(1, rel=3e-14)


@pytest.mark.parametrize("case", FIXTURE["tiny_coordinate_cases"])
def test_original_uninitialized_branch_can_lose_small_coordinates(case):
    _, _, _, f, dfn, dfd = case["input"]
    assert case["result"][0] == 0
    assert float(cum_f(f, dfn, dfd)) == pytest.approx(f / (1 + f), rel=3e-13, abs=0)


def test_broadcast_helpers_ownership_and_empty():
    f = np.array([[0.1], [1], [10.0]])
    dfn = np.array([0.2, 1, 5])
    r = cdf_f(f=f, dfn=dfn, dfd=2)
    np.testing.assert_array_equal(cum_f(f, dfn, 2), r.cum)
    np.testing.assert_array_equal(ccum_f(f, dfn, 2), r.ccum)
    np.testing.assert_allclose(inv_f(r.cum, dfn, 2, ccum=r.ccum), r.f)
    f[:] = 0
    dfn[:] = 1
    assert float(r.f[0, 0]) == 0.1 and float(r.dfn[0, 0]) == 0.2
    for name in ("cum", "ccum", "f", "dfn", "dfd"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert cdf_f(f=[], dfn=1, dfd=1).cum.size == 0
    assert inv_f([], 1, 1).size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (3, {}),
        (4, {}),
        (1.0, {}),
        (1, dict(f=1, dfn=1)),
        (1, dict(f=1, dfd=1)),
        (1, dict(dfn=1, dfd=1)),
        (1, dict(f=-1, dfn=1, dfd=1)),
        (1, dict(f=1e101, dfn=1, dfd=1)),
        (1, dict(f=1, dfn=0, dfd=1)),
        (1, dict(f=1, dfn=1, dfd=1e11)),
        (1, dict(f=1, dfn=1, dfd=1, cum=0.5)),
        (2, dict(cum=1, dfn=1, dfd=1)),
        (2, dict(ccum=1e-100, dfn=1, dfd=0.001)),
        (2, dict(cum=0.5, dfn=1, dfd=1, f=1)),
    ],
)
def test_invalid_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdf_f(which, **kwargs)


def test_strict_coordinate_bound():
    with pytest.raises(ValueError):
        cdf_f(f=np.nextafter(1e100, np.inf), dfn=1, dfd=1)
