"""Independent beta mixtures, closed forms, native contracts and boundaries."""

import json
from decimal import Decimal, localcontext
from math import exp, expm1, log, log1p
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_nc_f, cdf_f, cdf_nc_f, cum_nc_f, inv_nc_f

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_nc_f.json").read_text())


def beta_mixture(f, a, b, nc):
    with localcontext() as ctx:
        ctx.prec = 120
        f, mean = Decimal(str(f)), Decimal(str(nc)) / 2
        a = Decimal(str(a))
        z = a * f / (a * f + b)
        y = 1 - z
        weight = (-mean).exp()
        p = q = Decimal(0)
        for j in range(250):
            shape = a + j
            # I_z(a,b) = z**a * sum((a)_k * (1-z)**k / k!, k=0..b-1).
            term = total = Decimal(1)
            for k in range(1, b):
                term *= (shape + k - 1) * y / k
                total += term
            lower = z**shape * total
            p += weight * lower
            q += weight * (1 - lower)
            weight *= mean / (j + 1)
        assert weight < Decimal("1e-100")
        return float(p), float(q)


@pytest.mark.parametrize("a,b", [(1, 1), (1, 5), (5, 1), (2, 3)])
@pytest.mark.parametrize("f", [0.1, 1, 10])
@pytest.mark.parametrize("nc", [0.5, 4, 20])
def test_independent_decimal_poisson_beta_mixture(a, b, f, nc):
    p, q = beta_mixture(f, a, b, nc)
    r = cdf_nc_f(f=f, dfn=2 * a, dfd=2 * b, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(inv_nc_f(p, 2 * a, 2 * b, nc, ccum=q)) == pytest.approx(f, rel=3e-12)
    assert float(cdf_nc_f(3, cum=p, ccum=q, f=f, dfn=2 * a, dfd=2 * b).pnonc) == pytest.approx(
        nc, rel=3e-11
    )


@pytest.mark.parametrize("dfn", [0.001, 1, 20, 1e10])
@pytest.mark.parametrize("f", [1e-100, 0.1, 1, 10, 1e80])
def test_closed_form_denominator_two(dfn, f):
    nc = 4.0
    product = dfn * f
    z, y = product / (product + 2), 2 / (product + 2)
    logcdf = (dfn / 2) * (log1p(-y) if y <= 0.5 else log(z)) - (nc / 2) * y
    p, q = exp(logcdf), -expm1(logcdf)
    r = cdf_nc_f(f=f, dfn=dfn, dfd=2, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-12, atol=0)
    if p > 0 and q > 0:
        assert float(inv_nc_f(p, dfn, 2, nc, ccum=q)) == pytest.approx(f, rel=3e-10, abs=0)


@pytest.mark.parametrize("case", FIXTURE["profiles"]["central_status_repaired"]["cases"])
def test_native_repaired_reference(case):
    which, p, q, f, nn, dd, nc = case["input"]
    if which == 1:
        r = cdf_nc_f(f=f, dfn=nn, dfd=dd, pnonc=nc)
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=0, atol=1e-4)
    else:
        kwargs = dict(cum=p, ccum=q, f=f, dfn=nn, dfd=dd, pnonc=nc)
        name = "f" if which == 2 else "pnonc"
        truth = kwargs.pop(name)
        r = cdf_nc_f(which, **kwargs)
        if which == 3 and f == nn == dd == 10 and nc == 4:
            # Native probability truncation shifts this inverse by over .3%.
            assert abs(float(r.pnonc) / truth - 1) > 3e-3
        else:
            assert float(getattr(r, name)) == pytest.approx(truth, rel=3e-3, abs=1e-4)
        forward = cdf_nc_f(f=r.f, dfn=r.dfn, dfd=r.dfd, pnonc=r.pnonc)
        assert float(forward.cum if p <= q else forward.ccum) == pytest.approx(min(p, q), rel=3e-12)


def test_original_native_status_evidence():
    original = FIXTURE["profiles"]["original"]
    assert original["patches"] == []
    assert any(c["status"] == 10 for c in original["cases"])
    repaired = FIXTURE["profiles"]["central_status_repaired"]
    assert len(repaired["patches"]) == 1
    assert repaired["patches"][0]["file"] == "cdf_f_mod.f90"


@pytest.mark.parametrize(
    "nn,dd,nc,f", [(0.001, 2, 100, 1e5), (2, 0.001, 4, 1), (1e10, 1e10, 1e4, 1), (5, 10, 1e4, 2000)]
)
def test_parameter_bounds(nn, dd, nc, f):
    r = cdf_nc_f(f=f, dfn=nn, dfd=dd, pnonc=nc)
    assert float(inv_nc_f(r.cum, nn, dd, nc, ccum=r.ccum)) == pytest.approx(f, rel=3e-10)
    assert float(cdf_nc_f(3, cum=r.cum, ccum=r.ccum, f=f, dfn=nn, dfd=dd).pnonc) == pytest.approx(
        nc, rel=3e-10
    )


def test_central_reduction_and_mixed_zero_noncentrality():
    f = np.array([0.1, 1.0, 10.0])
    r = cdf_nc_f(f=f, dfn=2, dfd=2, pnonc=0)
    central = cdf_f(f=f, dfn=2, dfd=2)
    np.testing.assert_array_equal(r.cum, central.cum)
    np.testing.assert_array_equal(r.ccum, central.ccum)
    np.testing.assert_allclose(inv_nc_f(r.cum, 2, 2, 0, ccum=r.ccum), f, rtol=1e-13)
    r = cdf_nc_f(f=f, dfn=2, dfd=2, pnonc=[0, 4, 0])
    np.testing.assert_allclose(
        cdf_nc_f(3, cum=r.cum, ccum=r.ccum, f=f, dfn=2, dfd=2).pnonc, [0, 4, 0], rtol=1e-12, atol=0
    )


def test_zero_large_f_and_owned_broadcasts():
    f = np.array([[0.0], [1.0], [1e100]])
    nc = np.array([0.0, 4.0, 20.0])
    r = cdf_nc_f(f=f, dfn=2, dfd=2, pnonc=nc)
    np.testing.assert_array_equal(r.cum[0], 0)
    np.testing.assert_allclose(r.ccum[-1], (1 + nc / 2) * 1e-100, rtol=3e-13)
    np.testing.assert_array_equal(cum_nc_f(f, 2, 2, nc), r.cum)
    np.testing.assert_array_equal(ccum_nc_f(f, 2, 2, nc), r.ccum)
    f[:] = 1
    nc[:] = 1
    assert float(r.f[0, 0]) == 0 and float(r.pnonc[0, 0]) == 0
    for name in ("cum", "ccum", "f", "dfn", "dfd", "pnonc"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert float(inv_nc_f(0, 2, 2, 4)) == 0
    assert cdf_nc_f(f=[], dfn=2, dfd=2, pnonc=4).cum.size == 0
    assert inv_nc_f([], 2, 2, 4).size == 0
    assert cdf_nc_f(3, cum=[], f=[], dfn=2, dfd=2).pnonc.size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(f=1, dfn=1, dfd=1)),
        (1, dict(f=1, dfn=1, pnonc=1)),
        (1, dict(f=1, dfd=1, pnonc=1)),
        (1, dict(dfn=1, dfd=1, pnonc=1)),
        (1, dict(f=-1, dfn=1, dfd=1, pnonc=1)),
        (1, dict(f=1e101, dfn=1, dfd=1, pnonc=1)),
        (1, dict(f=1, dfn=0, dfd=1, pnonc=1)),
        (1, dict(f=1, dfn=1, dfd=1e11, pnonc=1)),
        (1, dict(f=1, dfn=1, dfd=1, pnonc=-1)),
        (1, dict(f=1, dfn=1, dfd=1, pnonc=1e5)),
        (1, dict(f=1, dfn=1, dfd=1, pnonc=1, cum=0.5)),
        (2, dict(f=1, dfn=1, dfd=1, pnonc=1, cum=0.5)),
        (2, dict(dfn=1, dfd=1, pnonc=1, cum=1)),
        (2, dict(dfn=1, dfd=0.001, pnonc=1, ccum=1e-100)),
        (3, dict(f=0, dfn=1, dfd=1, cum=0.5)),
        (3, dict(f=1, dfn=1, dfd=1, cum=1)),
        (3, dict(f=1, dfn=1, dfd=1, pnonc=1, cum=0.5)),
        (3, dict(f=0.1, dfn=2, dfd=2, cum=0.9)),
    ],
)
def test_invalid_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdf_nc_f(which, **kwargs)


def test_native_early_stopping_defect():
    inputs = [1, 0.0, 0.0, 0.1, 0.5, 10.0, 20.0]
    case = next(
        c for c in FIXTURE["profiles"]["central_status_repaired"]["cases"] if c["input"] == inputs
    )
    expected = beta_mixture(0.1, 0.25, 5, 20)
    assert case["result"][0] < expected[0] * 1e-10
    r = cdf_nc_f(f=0.1, dfn=0.5, dfd=10, pnonc=20)
    np.testing.assert_allclose([r.cum, r.ccum], expected, rtol=3e-13, atol=0)


def test_mixed_overflow_quantile_batch_preserves_other_rows():
    f = np.array([0.0, 1.0, 1e80])
    nn = np.array([2.0, 2.0, 1e10])
    r = cdf_nc_f(f=f, dfn=nn, dfd=2, pnonc=4)
    np.testing.assert_allclose(inv_nc_f(r.cum, nn, 2, 4, ccum=r.ccum), f, rtol=3e-12, atol=0)


def test_small_noncentrality_and_quantile_upper_bound():
    nc = 1e-11
    expected = 0.5 * exp(-nc / 4)
    assert float(cum_nc_f(1, 2, 2, nc)) == pytest.approx(expected, rel=1e-14)
    assert abs(float(cum_nc_f(1, 2, 2, nc)) - 0.5) > 1e-12
    r = cdf_nc_f(f=1e100, dfn=2, dfd=2, pnonc=4)
    assert float(inv_nc_f(r.cum, 2, 2, 4, ccum=r.ccum)) == pytest.approx(1e100, rel=1e-13)
