"""Legacy native contracts, independent mixtures, and nonmonotone df inversions."""

import json
from math import exp, expm1, lgamma, log, log1p
from pathlib import Path

import numpy as np
import pytest
from test_cdflib_nc_f import beta_mixture

from mdanderson_stats import cdf_nc_f, cdff, cdffnc, cumfnc

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_nc_f.json").read_text())
CASES = [(language, case) for language, p in FIXTURE["profiles"].items() for case in p["cases"]]


@pytest.mark.parametrize("language,case", CASES)
def test_native_c_and_fortran(language, case):
    which, p, q, f, nn, dd, nc = case["input"]
    kwargs = dict(f=f, dfn=nn, dfd=dd, pnonc=nc)
    if which != 1:
        kwargs.update(p=p, q=q)
        name = {2: "f", 3: "dfn", 4: "dfd", 5: "pnonc"}[which]
        truth = kwargs.pop(name)
        if which in (3, 4):
            kwargs["df_bracket"] = (truth * 0.95, truth * 1.05)
    result = cdffnc(which, **kwargs)
    if which == 1:
        # The archived summation is substantially less accurate than the
        # independent mixture checks below; retain its actual outputs.
        np.testing.assert_allclose([result.p, result.q], case["result"][:2], rtol=0, atol=2e-5)
    else:
        assert float(getattr(result, name)) == pytest.approx(truth, rel=0.004)
        lp, uq = cumfnc(result.f, result.dfn, result.dfd, result.pnonc)
        assert float(lp if p <= q else uq) == pytest.approx(min(p, q), rel=3e-12)
        if case["status"] == 0:
            _, _, nf, na, nb, noncentral, _ = case["result"]
            if which == 3 and (f, nn, dd, nc) == (0.1, 10, 0.5, 4):
                # A false native success, independently disproved by just
                # the first positive term of the j=0 beta-mixture component.
                a, b = na / 2, nb / 2
                z = na * nf / (na * nf + nb)
                bound = exp(
                    -noncentral / 2 + a * log(z) + lgamma(a + b) - lgamma(a + 1) - lgamma(b)
                )
                assert bound > 0.13 > p
                assert float(cumfnc(nf, na, nb, noncentral)[0]) >= bound
            else:
                np.testing.assert_allclose(
                    cumfnc(nf, na, nb, noncentral), [p, q], rtol=0, atol=2e-5
                )
    assert FIXTURE["profiles"][language]["adaptations"] == []


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_ignored_q_and_invalid_contracts(language):
    profile = FIXTURE["profiles"][language]
    assert [c["status"] for c in profile["invalid_cases"]] == [-1, -4, -5, -6, -7, -2, -2]
    native_f = []
    for case in profile["ignored_q_cases"]:
        _, p, q, _, nn, dd, nc = case["input"]
        assert case["status"] == 0
        assert case["result"][1] == q  # Native preserves the unused placeholder.
        native_f.append(case["result"][2])
        r = cdffnc(2, p=p, q=q, dfn=nn, dfd=dd, pnonc=nc)
        assert float(r.q) == 1 - p
        assert float(r.f) == pytest.approx(case["result"][2], rel=1e-5)
    assert len(set(native_f)) == 1


@pytest.mark.parametrize("a,b", [(1, 1), (1, 5), (5, 1), (2, 3)])
@pytest.mark.parametrize("f", [0.1, 1, 10])
@pytest.mark.parametrize("nc", [0.5, 4, 20])
def test_independent_decimal_mixture_all_inversions(a, b, f, nc):
    p, q = beta_mixture(f, a, b, nc)
    r = cdffnc(f=f, dfn=2 * a, dfd=2 * b, pnonc=nc)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    for which, name, truth in [(2, "f", f), (3, "dfn", 2 * a), (4, "dfd", 2 * b), (5, "pnonc", nc)]:
        kwargs = dict(p=p, f=f, dfn=2 * a, dfd=2 * b, pnonc=nc)
        kwargs.pop(name)
        if which in (3, 4):
            kwargs["df_bracket"] = (truth * 0.99, truth * 1.01)
        assert float(getattr(cdffnc(which, **kwargs), name)) == pytest.approx(truth, rel=3e-10)


@pytest.mark.parametrize("nn", [1e-100, 1e-4, 0.2, 2, 30, 1e12, 1e100])
@pytest.mark.parametrize("f", [0.1, 1, 10])
def test_independent_denominator_two_wide_domain(nn, f):
    nc = 4
    logp = -(nn / 2) * log1p(2 / (nn * f)) - nc / (nn * f + 2)
    p, q = exp(logp), -expm1(logp)
    r = cdffnc(f=f, dfn=nn, dfd=2, pnonc=nc)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-12, atol=0)
    # At extremely small nn the CDF becomes numerically independent of f.
    if nn >= 1e-4:
        assert float(cdffnc(2, p=p, dfn=nn, dfd=2, pnonc=nc).f) == pytest.approx(f, rel=3e-8)


@pytest.mark.parametrize(
    "which,f,fixed,nc,p,turn", [(3, 5, 5, 0.5, 0.77, 0.045), (4, 0.5, 10, 4, 0.1, 0.65)]
)
def test_two_df_roots_can_be_selected(which, f, fixed, nc, p, turn):
    kwargs = {"f": f, "pnonc": nc, "p": p, "dfd" if which == 3 else "dfn": fixed}
    with pytest.raises(ValueError, match="df_bracket"):
        cdffnc(which, **kwargs)
    r = cdffnc(which, **kwargs, df_bracket=([0.001, turn], [turn, 100]))
    roots = r.dfn if which == 3 else r.dfd
    assert roots[0] < turn < roots[1]
    np.testing.assert_allclose(cumfnc(r.f, r.dfn, r.dfd, r.pnonc)[0], p, rtol=2e-12)
    for root in roots:
        varied = root * np.array([0.999, 1.001])
        lp = cumfnc(f, varied if which == 3 else fixed, fixed if which == 3 else varied, nc)[0]
        assert (lp[0] - p) * (lp[1] - p) < 0


@pytest.mark.parametrize("which", [3, 4])
def test_mixed_endpoint_and_interior_df_batch(which):
    truth = np.array([[2.0, 3.0, 4.0], [6.0, 8.0, 10.0]])[:, ::-1]
    f, nc, fixed = np.array([[0.7], [1.3]]), np.array([0.0, 0.5, 4.0]), np.array([7, 9, 11])
    nn, dd = (truth, fixed) if which == 3 else (fixed, truth)
    r = cdffnc(f=f, dfn=nn, dfd=dd, pnonc=nc)
    low, high = truth * 0.99, truth * 1.01
    low[0, 0], high[1, 1] = truth[0, 0], truth[1, 1]
    kwargs = {"dfd" if which == 3 else "dfn": fixed}
    result = cdffnc(which, p=r.p, f=f, pnonc=nc, df_bracket=(low, high), **kwargs)
    actual = result.dfn if which == 3 else result.dfd
    assert actual[0, 0] == truth[0, 0] and actual[1, 1] == truth[1, 1]
    np.testing.assert_allclose(actual, truth, rtol=3e-11)


def test_wide_inputs_central_reduction_and_noncentrality_bounds():
    for nn, dd in [(1e200, 2), (2, 1e200), (1e200, 1e200)]:
        r = cdffnc(f=1, dfn=nn, dfd=dd, pnonc=0)
        central = cdff(f=1, dfn=nn, dfd=dd)
        np.testing.assert_array_equal([r.p, r.q], [central.p, central.q])
        assert float(cdffnc(2, p=r.p, dfn=nn, dfd=dd, pnonc=0).f) == pytest.approx(1, rel=3e-12)
    r = cdffnc(f=1e200, dfn=2, dfd=2, pnonc=4)
    assert float(r.q) == pytest.approx(3e-200, rel=3e-12, abs=0)
    for nc in [0, 1e4]:
        r = cdffnc(f=2000, dfn=5, dfd=10, pnonc=nc)
        assert float(cdffnc(5, p=r.p, f=r.f, dfn=r.dfn, dfd=r.dfd).pnonc) == nc
    # An input noncentrality above the inverse bound is permitted.
    r = cdffnc(f=1e5, dfn=2, dfd=2, pnonc=1e5)
    logp = -log1p(1e-5) - 1e5 / (2e5 + 2)
    np.testing.assert_allclose([r.p, r.q], [exp(logp), -expm1(logp)], rtol=3e-11)


def test_ignored_q_shape_probability_limit_and_ownership():
    r = cdffnc(2, p=[0, 0.4, 1 - 1e-16], q=np.full((7, 9), np.nan), dfn=2, dfd=2, pnonc=1)
    assert r.f.shape == (3,) and r.f[0] == 0
    np.testing.assert_allclose(cumfnc(r.f, r.dfn, r.dfd, r.pnonc)[0], r.p, rtol=3e-13)
    f = np.array([[0.0], [1.0]])
    r = cdffnc(f=f, dfn=2, dfd=2, pnonc=[0, 4, 20])
    f[:] = 2
    assert r.f.shape == (2, 3) and r.f[0, 0] == 0
    for name in ("p", "q", "f", "dfn", "dfd", "pnonc"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which, name in [(1, "p"), (2, "f"), (3, "dfn"), (4, "dfd"), (5, "pnonc")]:
        kwargs = dict(f=[], dfn=[], dfd=[], pnonc=[])
        if which != 1:
            kwargs.pop(name)
            kwargs["p"] = []
        assert getattr(cdffnc(which, **kwargs), name).size == 0


def test_f95_overlap_and_distinct_mode_numbers():
    r = cdf_nc_f(f=[0.1, 1, 10], dfn=2, dfd=10, pnonc=4)
    legacy = cdffnc(f=r.f, dfn=r.dfn, dfd=r.dfd, pnonc=r.pnonc)
    np.testing.assert_array_equal([legacy.p, legacy.q], [r.cum, r.ccum])
    np.testing.assert_allclose(cdffnc(5, p=r.cum, f=r.f, dfn=r.dfn, dfd=r.dfd).pnonc, 4, rtol=3e-12)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (6, {}),
        (1.0, {}),
        (1, dict(f=1, dfn=2, dfd=2)),
        (1, dict(f=-1, dfn=2, dfd=2, pnonc=1)),
        (1, dict(f=1, dfn=0, dfd=2, pnonc=1)),
        (1, dict(f=1, dfn=2, dfd=0, pnonc=1)),
        (1, dict(f=1, dfn=2, dfd=2, pnonc=-1)),
        (1, dict(f=np.inf, dfn=2, dfd=2, pnonc=1)),
        (1, dict(f=1, dfn=2, dfd=2, pnonc=np.nan)),
        (1, dict(f=1, dfn=2, dfd=2, pnonc=1, q=0.5)),
        (2, dict(p=-0.1, dfn=2, dfd=2, pnonc=1)),
        (2, dict(p=1, dfn=2, dfd=2, pnonc=1)),
        (2, dict(q=0.5, dfn=2, dfd=2, pnonc=1)),
        (2, dict(p=0.5, dfn=2, dfd=2, pnonc=1, df_bracket=(1, 2))),
        (2, dict(p=0.5, dfn=2, dfd=0.001, pnonc=1)),
        (3, dict(p=0.5, f=0, dfd=2, pnonc=1)),
        (3, dict(p=0, f=1, dfd=2, pnonc=1)),
        (3, dict(p=0.5, f=1, dfd=2, pnonc=1, df_bracket=(0, 1))),
        (3, dict(p=0.5, f=1, dfd=2, pnonc=1, df_bracket=(1, 1e101))),
        (4, dict(p=0.5, f=1, dfn=2, pnonc=1, df_bracket=(2, 1))),
        (4, dict(p=0.5, f=1, dfn=2, pnonc=1, df_bracket=(1,))),
        (5, dict(p=0, f=1, dfn=2, dfd=2)),
        (5, dict(p=0.9, f=0.1, dfn=2, dfd=2)),
    ],
)
def test_invalid_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdffnc(which, **kwargs)


@pytest.mark.parametrize(
    "nn,f,nc",
    [
        (1e308, 1e308, 1e308),
        (np.nextafter(0.0, 1.0), 1e-100, 1e-320),
        (1e308, np.nextafter(0.0, 1.0), 4),
    ],
)
def test_denominator_two_extreme_scaling_against_decimal(nn, f, nc):
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 800
        n, x, noncentral = map(Decimal.from_float, (float(nn), float(f), float(nc)))
        logp = -(n / 2) * (1 + 2 / (n * x)).ln() - noncentral / (n * x + 2)
        p = logp.exp()
        expected = [float(p), float(1 - p)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = cumfnc(f, nn, 2, nc)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=8 * np.nextafter(0.0, 1.0))
