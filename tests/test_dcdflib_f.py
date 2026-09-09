"""Both legacy languages, independent F identities and wider-domain inversions."""

import json
from math import exp, expm1, log1p
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_f, cdff, cumf

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_f.json").read_text())
CASES = [(language, case) for language, p in FIXTURE["profiles"].items() for case in p["cases"]]


@pytest.mark.parametrize("language,case", CASES)
def test_native_c_and_fortran(language, case):
    which, p, q, f, nn, dd = case["input"]
    kwargs = dict(f=f, dfn=nn, dfd=dd)
    if which != 1:
        kwargs.update(p=p, q=q)
        name = {2: "f", 3: "dfn", 4: "dfd"}[which]
        truth = kwargs.pop(name)
        if which in (3, 4):
            kwargs["df_bracket"] = (truth * 0.99, truth * 1.01)
    r = cdff(which, **kwargs)
    if which == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=5e-12, atol=0)
    else:
        assert float(getattr(r, name)) == pytest.approx(truth, rel=3e-8)
        if case["status"] == 0:
            _, _, nf, na, nb, _ = case["result"]
            lp, uq = cumf(nf, na, nb)
            assert float(lp if p <= q else uq) == pytest.approx(min(p, q), rel=2e-7, abs=1e-15)
    assert FIXTURE["profiles"][language]["adaptations"] == []


@pytest.mark.parametrize("nn", [1e-100, 1e-20, 0.001, 1, 1e10, 1e50, 1e100])
@pytest.mark.parametrize("f", [0.1, 1, 10])
def test_independent_denominator_two_and_reciprocal(nn, f):
    logp = -(nn / 2) * log1p(2 / (nn * f))
    p, q = exp(logp), -expm1(logp)
    r = cdff(f=f, dfn=nn, dfd=2)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    assert float(cdff(2, p=p, q=q, dfn=nn, dfd=2).f) == pytest.approx(f, rel=3e-12)
    back = cdff(f=1 / f, dfn=2, dfd=nn)
    np.testing.assert_allclose([back.p, back.q], [q, p], rtol=3e-13, atol=0)
    assert float(cdff(2, p=q, q=p, dfn=2, dfd=nn).f) == pytest.approx(1 / f, rel=3e-12)


@pytest.mark.parametrize("nn,dd", [(1e-100, 2), (1e100, 2), (2, 1e-100), (2, 1e100), (1e100, 100)])
def test_legacy_search_bounds_and_local_bracketing(nn, dd):
    r = cdff(f=1, dfn=nn, dfd=dd)
    assert float(cdff(3, p=r.p, q=r.q, f=1, dfd=dd).dfn) == pytest.approx(nn, rel=3e-11)
    assert float(cdff(4, p=r.p, q=r.q, f=1, dfn=nn).dfd) == pytest.approx(dd, rel=3e-11)


@pytest.mark.parametrize("which,f,fixed,p,turn", [(3, 5, 5, 0.94, 0.43), (4, 0.5, 10, 0.15, 1.33)])
def test_both_df_roots_selectable(which, f, fixed, p, turn):
    kwargs = {"f": f, "p": p, "dfd" if which == 3 else "dfn": fixed}
    with pytest.raises(ValueError, match="df_bracket"):
        cdff(which, **kwargs)
    r = cdff(which, **kwargs, df_bracket=([0.001, turn], [turn, 100]))
    roots = r.dfn if which == 3 else r.dfd
    assert roots[0] < turn < roots[1]
    lp, uq = cumf(r.f, r.dfn, r.dfd)
    np.testing.assert_allclose(lp, p, rtol=1e-12)
    for root in roots:
        varied = root * np.array([0.999, 1.001])
        probability = cumf(f, varied if which == 3 else fixed, fixed if which == 3 else varied)[0]
        assert (probability[0] - p) * (probability[1] - p) < 0


def test_input_domains_exceed_search_bounds_and_scaled_coordinates():
    # Source input validation has no upper df/f bound; search bounds differ.
    for nn, dd in [(1e200, 1e200), (1e200, 2)]:
        r = cdff(f=1, dfn=nn, dfd=dd)
        assert float(cdff(2, p=r.p, q=r.q, dfn=nn, dfd=dd).f) == pytest.approx(1, rel=3e-13)
    assert float(cdff(f=1e200, dfn=2, dfd=2).q) == pytest.approx(1e-200, rel=3e-13, abs=0)
    assert float(cdff(f=1e200, dfn=1e200, dfd=1e200).p) == 1
    r = cdff(f=1e-300, dfn=1e-100, dfd=1e-100)
    assert 0 < float(r.p) < 1
    assert float(cdff(f=1, dfn=1e308, dfd=1e308).p) == 0.5
    with pytest.raises(ArithmeticError, match="underflowed"):
        cdff(f=1e308, dfn=1e308, dfd=1e-308)


def test_zero_upper_bound_broadcasts_and_f95_overlap():
    f = np.array([[0.0], [1.0], [1e100]])
    nn = np.array([1.0, 2.0, 5.0])
    r = cdff(f=f, dfn=nn, dfd=2)
    expected = cdf_f(f=f, dfn=nn, dfd=2)
    np.testing.assert_allclose([r.p, r.q], [expected.cum, expected.ccum], rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdff(2, p=r.p, q=r.q, dfn=nn, dfd=2).f, r.f, rtol=3e-13, atol=0)
    f[:] = 1
    nn[:] = 2
    assert float(r.f[0, 0]) == 0 and float(r.dfn[0, 0]) == 1
    for name in ("p", "q", "f", "dfn", "dfd"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert cdff(f=[], dfn=2, dfd=2).p.size == 0
    assert cdff(2, p=[], dfn=2, dfd=2).f.size == 0
    assert cdff(3, p=[], f=[], dfd=2).dfn.size == 0
    assert cdff(4, p=[], f=[], dfn=2).dfd.size == 0


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_invalid_contracts(language):
    cases = FIXTURE["profiles"][language]["invalid_cases"]
    assert [c["status"] for c in cases] == [-1, -4, -5, -6, -3, 3]
    for case in cases:
        w, p, q, f, nn, dd = case["input"]
        assert case["status"] != 0
        kwargs = dict(f=f, dfn=nn, dfd=dd)
        if w == 2:
            kwargs.pop("f")
            kwargs.update(p=p, q=q)
        with pytest.raises(ValueError):
            cdff(w, **kwargs)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(f=1, dfn=1)),
        (1, dict(f=1, dfd=1)),
        (1, dict(dfn=1, dfd=1)),
        (1, dict(f=1, dfn=1, dfd=1, p=0.5)),
        (2, dict(f=1, dfn=1, dfd=1, p=0.5)),
        (3, dict(f=1, dfn=1, dfd=1, p=0.5)),
        (4, dict(f=1, dfn=1, dfd=1, p=0.5)),
        (3, dict(f=0, dfd=1, p=0.5)),
        (4, dict(f=1, dfn=1, p=0)),
        (3, dict(f=1, dfd=1, p=0.5, df_bracket=(1e-101, 2))),
        (4, dict(f=1, dfn=1, p=0.5, df_bracket=(1, 1e101))),
        (3, dict(f=1, dfd=1, p=0.5, df_bracket=(2, 1))),
        (1, dict(f=1, dfn=1, dfd=1, df_bracket=(1, 2))),
        (2, dict(dfn=2, dfd=2, p=0.5, q=0.5 + 4 * np.finfo(float).eps)),
        (2, dict(dfn=2, dfd=2, q=1e-200)),
    ],
)
def test_invalid_unidentified_and_out_of_bounds(which, kwargs):
    with pytest.raises(ValueError):
        cdff(which, **kwargs)
