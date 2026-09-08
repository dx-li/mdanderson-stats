"""Native negative-binomial outputs and independent binomial/geometric identities."""

import json
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_neg_binomial, cdf_neg_binomial, cum_neg_binomial, inv_neg_binomial

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_neg_binomial.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_reference(case):
    which, p, q, f, s, pr, cpr = case["input"]
    kwargs = dict(cum=p, ccum=q, f=f, s=s, pr=pr, cpr=cpr)
    for name in {1: ("cum", "ccum"), 2: ("f",), 3: ("s",), 4: ("pr", "cpr")}[which]:
        kwargs.pop(name)
    r = cdf_neg_binomial(which, **kwargs)
    if which == 1:
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        np.testing.assert_allclose([r.f, r.s, r.pr, r.cpr], [f, s, pr, cpr], rtol=3e-11, atol=1e-13)


@pytest.mark.parametrize("f", [0, 1, 5])
@pytest.mark.parametrize("s", [1, 2, 5])
@pytest.mark.parametrize("pr", [Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)])
def test_exact_rational_binomial_sum(f, s, pr):
    n = f + s
    p = sum(Fraction(comb(n, k)) * pr**k * (1 - pr) ** (n - k) for k in range(s, n + 1))
    pp, qq = float(p), float(1 - p)
    r = cdf_neg_binomial(f=f, s=s, pr=float(pr))
    np.testing.assert_allclose([r.cum, r.ccum], [pp, qq], rtol=3e-13, atol=0)
    assert float(inv_neg_binomial(pp, s, float(pr), ccum=qq)) == pytest.approx(
        f, rel=2e-12, abs=1e-13
    )
    assert float(cdf_neg_binomial(3, f=f, pr=float(pr), cum=pp, ccum=qq).s) == pytest.approx(
        s, rel=2e-12
    )
    assert float(cdf_neg_binomial(4, f=f, s=s, cum=pp, ccum=qq).pr) == pytest.approx(
        float(pr), rel=2e-13
    )


@pytest.mark.parametrize("s", [1e-300, 1e-100, 1e-20, 0.2, 1, 5])
def test_zero_failure_power_identity_and_tiny_success_counts(s):
    pr = 0.3
    p, q = np.exp(s * np.log(pr)), -np.expm1(s * np.log(pr))
    r = cdf_neg_binomial(f=0, s=s, pr=pr)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(cdf_neg_binomial(3, f=0, pr=pr, cum=p, ccum=q).s) == pytest.approx(
        s, rel=3e-12, abs=0
    )


@pytest.mark.parametrize("f", [0.0, 0.2, 3.5, 1e10])
def test_geometric_identity_and_upper_failure_bound(f):
    cpr = 0.8 if f < 1e10 else np.exp(-1 / (f + 1))
    p, q = -np.expm1((f + 1) * np.log(cpr)), np.exp((f + 1) * np.log(cpr))
    r = cdf_neg_binomial(f=f, s=1, cpr=cpr)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=2e-12, atol=0)
    assert float(inv_neg_binomial(p, 1, None, cpr=cpr, ccum=q)) == pytest.approx(
        f, rel=2e-12, abs=1e-13
    )


def test_zero_successes_and_probability_endpoints():
    r = cdf_neg_binomial(f=[0, 1, 3], s=0, pr=[0, 0.5, 1])
    np.testing.assert_array_equal(r.cum, [1, 1, 1])
    np.testing.assert_array_equal(r.ccum, [0, 0, 0])
    np.testing.assert_array_equal(cdf_neg_binomial(3, f=[0, 1], pr=[0.2, 0.8], cum=1).s, [0, 0])
    r = cdf_neg_binomial(4, f=3, s=2, cum=[0, 1])
    np.testing.assert_array_equal(r.pr, [0, 1])
    np.testing.assert_array_equal(r.cpr, [1, 0])
    r = cdf_neg_binomial(f=3, s=2, pr=[0, 1])
    np.testing.assert_array_equal(r.cum, [0, 1])


def test_extreme_complements_broadcast_ownership_and_empty():
    r = cdf_neg_binomial(4, f=0, s=1, ccum=1e-100)
    assert float(r.pr) == 1 and float(r.cpr) == pytest.approx(1e-100, rel=1e-13)
    f = np.array([[0.5], [2.0]])
    s = np.array([0.2, 1, 3])
    r = cdf_neg_binomial(f=f, s=s, pr=0.4)
    np.testing.assert_array_equal(cum_neg_binomial(f, s, 0.4), r.cum)
    np.testing.assert_array_equal(ccum_neg_binomial(f, s, 0.4), r.ccum)
    np.testing.assert_allclose(inv_neg_binomial(r.cum, s, 0.4, ccum=r.ccum), r.f)
    f[:] = 0
    s[:] = 0
    assert float(r.f[0, 0]) == 0.5 and float(r.s[0, 0]) == 0.2
    for name in ("cum", "ccum", "f", "s", "pr", "cpr"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which in (1, 2, 3, 4):
        kwargs = dict(cum=[], ccum=[], f=[], s=[], pr=[], cpr=[])
        for name in {1: ("cum", "ccum"), 2: ("f",), 3: ("s",), 4: ("pr", "cpr")}[which]:
            kwargs.pop(name)
        assert cdf_neg_binomial(which, **kwargs).cum.size == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(f=1, pr=0.5)),
        (1, dict(s=1, pr=0.5)),
        (1, dict(f=-1, s=1, pr=0.5)),
        (1, dict(f=1, s=-1, pr=0.5)),
        (1, dict(f=1, s=1e11, pr=0.5)),
        (1, dict(f=1, s=1, pr=0.5, cpr=0.6)),
        (1, dict(f=1, s=1, pr=np.nan)),
        (2, dict(s=0, pr=0.5, cum=0.5)),
        (2, dict(s=1, pr=0.5, cum=0.1)),
        (2, dict(s=1, pr=0.5, cum=1)),
        (2, dict(s=1, pr=0, cum=0.5)),
        (3, dict(f=1, pr=0.5, cum=0)),
        (3, dict(f=1, pr=1, cum=0.5)),
        (4, dict(f=1, s=0, cum=1)),
        (4, dict(f=1, s=1, cum=0.5, pr=0.5)),
    ],
)
def test_invalid_or_unidentified_requests(which, kwargs):
    with pytest.raises(ValueError):
        cdf_neg_binomial(which, **kwargs)


@pytest.mark.parametrize("s", [1e10, np.nextafter(1e10, 0)])
def test_success_upper_bound(s):
    r = cdf_neg_binomial(f=0, s=s, cpr=1e-10)
    answer = cdf_neg_binomial(3, f=0, cpr=1e-10, cum=r.cum, ccum=r.ccum)
    assert float(answer.s) == pytest.approx(s, rel=2e-13)


def test_mixed_zero_success_inverse_keeps_other_inputs():
    result = cdf_neg_binomial(3, f=[2, 0], pr=[0.4, 0.5], cum=[1, 0.25])
    np.testing.assert_allclose(result.s, [0, 2], rtol=2e-13)
    np.testing.assert_array_equal(result.f, [2, 0])
    np.testing.assert_array_equal(result.pr, [0.4, 0.5])


def test_strict_count_upper_bounds():
    for name in ("f", "s"):
        kwargs = dict(f=0, s=1, pr=0.5)
        kwargs[name] = np.nextafter(1e10, np.inf)
        with pytest.raises(ValueError):
            cdf_neg_binomial(**kwargs)
