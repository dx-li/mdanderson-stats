"""Legacy Poisson native evidence and independent finite-sum identities."""

import json
from decimal import Decimal, localcontext
from math import erfc, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_poisson, cdfpoi, cumpoi

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_poisson.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]
TINY = np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("language,case", CASES)
def test_unchanged_native_references(language, case):
    w, p, q, s, mean = case["input"]
    kw = dict(s=s, mean=mean)
    if w != 1:
        kw.update(p=p, q=q)
        name = {2: "s", 3: "mean"}[w]
        truth = kw.pop(name)
    result = cdfpoi(w, **kw)
    assert case["status"] == 0
    if w == 1:
        np.testing.assert_allclose([result.p, result.q], case["result"][:2], rtol=3e-12, atol=0)
    else:
        assert float(getattr(result, name)) == pytest.approx(truth, rel=3e-7, abs=1e-14)
        _, _, ns, nm, _ = case["result"]
        native = cdfpoi(s=ns, mean=nm)
        np.testing.assert_allclose([native.p, native.q], [p, q], rtol=3e-7, atol=1e-15)
    assert FIXTURE["profiles"][language]["adaptations"] == []


def finite_sum(s, mean, precision=120):
    with localcontext() as ctx:
        ctx.prec = precision
        mu = Decimal.from_float(float(mean))
        term = total = Decimal(1)
        for k in range(1, s + 1):
            term *= mu / k
            total += term
        p = (-mu).exp() * total
        return float(p), float(1 - p)


@pytest.mark.parametrize("s", [0, 1, 5, 20])
@pytest.mark.parametrize("mean", [0.001, 0.1, 1, 10, 100])
def test_independent_decimal_finite_sum_all_modes(s, mean):
    p, q = finite_sum(s, mean)
    r = cdfpoi(s=s, mean=mean)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdfpoi(2, p=p, q=q, mean=mean).s, s, rtol=3e-11, atol=1e-14)
    np.testing.assert_allclose(cdfpoi(3, p=p, q=q, s=s).mean, mean, rtol=3e-11, atol=0)


@pytest.mark.parametrize("s", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("mean", [0.1, 1, 10])
def test_independent_half_integer_recurrence(s, mean):
    p = erfc(sqrt(mean))
    term = 2 * sqrt(mean) * exp(-mean) / sqrt(pi)
    for n in range(int(s + 0.5)):
        p += term
        term *= mean / (n + 1.5)
    r = cdfpoi(s=s, mean=mean)
    np.testing.assert_allclose([r.p, r.q], [p, 1 - p], rtol=3e-11, atol=3e-15)
    np.testing.assert_allclose(cdfpoi(2, p=p, mean=mean).s, s, rtol=3e-9)
    np.testing.assert_allclose(cdfpoi(3, p=p, s=s).mean, mean, rtol=3e-9)


@pytest.mark.parametrize("s", [0, 1, 2])
@pytest.mark.parametrize("mean", [TINY, 1e-160, 1e-100, 1e-50])
def test_independent_800_digit_small_mean(s, mean):
    p, q = finite_sum(s, mean, 800)
    r = cdfpoi(s=s, mean=mean)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-13, atol=TINY)
    if q > 0:
        inv = cdfpoi(3, p=p, q=q, s=s)
        np.testing.assert_allclose(cdfpoi(s=s, mean=inv.mean).q, q, rtol=3e-12, atol=TINY)
        if q >= np.finfo(float).tiny:
            np.testing.assert_allclose(inv.mean, mean, rtol=3e-12, atol=0)
        np.testing.assert_allclose(cdfpoi(2, p=p, q=q, mean=mean).s, s, rtol=3e-6, atol=1e-14)


def test_native_failures_zero_mean_and_wide_boundaries():
    for profile in FIXTURE["profiles"].values():
        assert [c["status"] for c in profile["invalid_cases"]] == [-1, -4, -5, -2, -3, 3]
        wide = profile["wide_cases"]
        assert wide[0]["status"] == 0 and wide[0]["result"][:2] == ["nan", "nan"]
        assert wide[1]["status"] == 0 and wide[1]["result"][:2] == [0, 2]
        assert wide[2]["status"] == 1 and wide[3]["status"] == 2
        assert wide[5]["status"] == 0 and wide[5]["result"][3] == 0
        assert wide[7]["status"] == 0 and wide[7]["result"][3] > 1000
    np.testing.assert_allclose(cdfpoi(3, q=1e-100, s=0).mean, 1e-100, rtol=3e-13, atol=0)
    for large in (1e100, 1e200, 1e308):
        r = cdfpoi(s=large, mean=large)
        assert float(r.p) == 0.5 and float(r.q) == 0.5
    for w, kw in [(2, dict(mean=1e200)), (3, dict(s=1e200))]:
        with pytest.raises(ValueError):
            cdfpoi(w, p=0.5, **kw)
    s = np.array([0, 0.5, 3, 1e90, 1e100])
    mean = np.array([TINY, 1, 2, 1e90, 1e100])
    r = cdfpoi(s=s, mean=mean)
    np.testing.assert_allclose(cdfpoi(2, p=r.p, q=r.q, mean=mean).s, s, rtol=3e-12, atol=0)
    assert float(cdfpoi(3, p=0.5, s=1e100).mean) == 1e100
    zero = cdfpoi(s=[0, 0.5, 1e308], mean=0)
    np.testing.assert_array_equal(zero.p, 1)
    np.testing.assert_array_equal(zero.q, 0)


def test_broadcast_owned_empty_and_f95_overlap():
    s = np.array([[0.0], [1.0], [10.0]])
    mean = np.array([0.2, 1, 10.0])
    r = cdfpoi(s=s, mean=mean)
    f95 = cdf_poisson(s=s, mean=mean)
    np.testing.assert_allclose([r.p, r.q], [f95.cum, f95.ccum], rtol=3e-14, atol=0)
    s[:] = 8
    mean[:] = 8
    assert r.s[0, 0] == 0 and r.mean[0, 0] == 0.2
    for name in ["p", "q", "s", "mean"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "s"), (3, "mean")]:
        kw = dict(s=[], mean=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfpoi(w, **kw), name).size == 0
    np.testing.assert_allclose(cumpoi(0, 1), [exp(-1), 1 - exp(-1)], rtol=3e-14)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (1.0, {}),
        (1, dict(s=1)),
        (1, dict(mean=1)),
        (1, dict(s=-1, mean=2)),
        (1, dict(s=1, mean=-1)),
        (1, dict(s=np.inf, mean=2)),
        (1, dict(s=1, mean=np.nan)),
        (1, dict(s=1, mean=2, p=0.5)),
        (2, dict(q=0, mean=2)),
        (2, dict(p=0.4, q=0.4, mean=2)),
        (2, dict(p=0.5, q=0.5 + 4 * np.finfo(float).eps, mean=2)),
        (2, dict(p=0.5, mean=0)),
        (2, dict(p=0.1, mean=0.1)),
        (2, dict(p=0, mean=1)),
        (3, dict(p=0, s=0)),
        (3, dict(q=0, s=0)),
        (3, dict(p=0.5, s=1, mean=1)),
    ],
)
def test_invalid_unidentified_unrepresentable(which, kwargs):
    with pytest.raises(ValueError):
        cdfpoi(which, **kwargs)
