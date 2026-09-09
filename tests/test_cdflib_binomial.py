"""Native binomial references, exact sums and continuous count identities."""

import json
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_binomial, cdf_binomial, cum_binomial, inv_binomial

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_binomial.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_binomial(case):
    which, p, q, s, n, pr, cpr = case["input"]
    kwargs = dict(cum=p, ccum=q, s=s, n=n, pr=pr, cpr=cpr)
    for name in {1: ("cum", "ccum"), 2: ("s",), 3: ("n",), 4: ("pr", "cpr")}[which]:
        kwargs.pop(name)
    r = cdf_binomial(which, **kwargs)
    if which == 1:
        assert case["status"] == 0
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        np.testing.assert_allclose([r.s, r.n, r.pr, r.cpr], [s, n, pr, cpr], rtol=3e-11, atol=1e-13)


@pytest.mark.parametrize("n", [1, 5, 20])
@pytest.mark.parametrize("pr", [Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)])
def test_exact_integer_binomial_sums(n, pr):
    for s in range(n + 1):
        p = sum(Fraction(comb(n, k)) * pr**k * (1 - pr) ** (n - k) for k in range(s + 1))
        pp, qq = float(p), float(1 - p)
        r = cdf_binomial(s=s, n=n, pr=float(pr))
        np.testing.assert_allclose([r.cum, r.ccum], [pp, qq], rtol=3e-13, atol=0)
        assert float(inv_binomial(pp, n, float(pr), ccum=qq)) == pytest.approx(
            s, rel=2e-12, abs=1e-13
        )
        assert float(cdf_binomial(3, s=s, pr=float(pr), cum=pp, ccum=qq).n) == pytest.approx(
            n, rel=2e-12
        )
        if s < n:
            assert float(cdf_binomial(4, s=s, n=n, cum=pp, ccum=qq).pr) == pytest.approx(
                float(pr), rel=2e-13
            )


@pytest.mark.parametrize("n", [1e-300, 1e-100, 1e-20, 0.2, 1, 5])
def test_zero_success_power_identity(n):
    pr = 0.3
    p, q = np.exp(n * np.log1p(-pr)), -np.expm1(n * np.log1p(-pr))
    r = cdf_binomial(s=0, n=n, pr=pr)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(cdf_binomial(3, s=0, pr=pr, cum=p, ccum=q).n) == pytest.approx(n, rel=3e-12, abs=0)
    assert float(inv_binomial(p, n, pr, ccum=q)) == 0


@pytest.mark.parametrize("s,n", [(0.2, 1), (0.5, 3.5), (1.7, 5), (5.2, 20)])
def test_fractional_counts_roundtrip(s, n):
    r = cdf_binomial(s=s, n=n, pr=0.4)
    assert float(inv_binomial(r.cum, n, 0.4, ccum=r.ccum)) == pytest.approx(s, rel=3e-12)
    assert float(cdf_binomial(3, s=s, pr=0.4, cum=r.cum, ccum=r.ccum).n) == pytest.approx(
        n, rel=3e-12
    )


def test_boundaries_and_unique_degenerate_solutions():
    np.testing.assert_array_equal(
        cdf_binomial(s=[0, 2, 5], n=[0, 2, 5], pr=[0, 0.5, 1]).cum, [1, 1, 1]
    )
    np.testing.assert_array_equal(cdf_binomial(s=1, n=3, pr=[0, 1]).cum, [1, 0])
    np.testing.assert_array_equal(cdf_binomial(4, s=1, n=3, cum=[0, 1]).pr, [1, 0])
    np.testing.assert_array_equal(inv_binomial(1, [0, 2], [0, 1]), [0, 2])
    np.testing.assert_array_equal(cdf_binomial(3, s=[1, 1e10], pr=[1, 0], cum=1).n, [1, 1e10])
    for s in (0, 1e10 / 2):
        r = cdf_binomial(s=s, n=1e10, pr=0.5 if s else 1e-10)
        assert float(cdf_binomial(3, s=s, pr=r.pr, cum=r.cum, ccum=r.ccum).n) == pytest.approx(1e10)


def test_broadcast_immutable_empty_and_complements():
    s = np.array([[0.2], [0.5]])
    n = np.array([1, 3, 5])
    r = cdf_binomial(s=s, n=n, pr=0.4)
    np.testing.assert_array_equal(cum_binomial(s, n, 0.4), r.cum)
    np.testing.assert_array_equal(ccum_binomial(s, n, 0.4), r.ccum)
    np.testing.assert_allclose(inv_binomial(r.cum, n, 0.4, ccum=r.ccum), r.s)
    s[:] = 0
    n[:] = 0
    assert float(r.s[0, 0]) == 0.2 and float(r.n[0, 0]) == 1
    for name in ("cum", "ccum", "s", "n", "pr", "cpr"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which in (1, 2, 3, 4):
        kwargs = dict(cum=[], ccum=[], s=[], n=[], pr=[], cpr=[])
        for name in {1: ("cum", "ccum"), 2: ("s",), 3: ("n",), 4: ("pr", "cpr")}[which]:
            kwargs.pop(name)
        assert cdf_binomial(which, **kwargs).cum.size == 0
    r = cdf_binomial(4, s=0, n=1, cum=1e-100)
    assert float(r.pr) == 1 and float(r.cpr) == pytest.approx(1e-100, rel=1e-13)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(s=0, pr=0.5)),
        (1, dict(n=1, pr=0.5)),
        (1, dict(s=-1, n=1, pr=0.5)),
        (1, dict(s=2, n=1, pr=0.5)),
        (1, dict(s=0, n=1e11, pr=0.5)),
        (1, dict(s=0, n=1, pr=0.5, cpr=0.6)),
        (1, dict(s=0, n=1, pr=np.nan)),
        (2, dict(n=1, pr=0.5, cum=0.1)),
        (2, dict(n=1, pr=0, cum=1)),
        (2, dict(n=1, pr=1, cum=0)),
        (2, dict(n=0, pr=0.5, cum=0.5)),
        (3, dict(s=1, pr=0, cum=1)),
        (3, dict(s=1, pr=1, cum=0)),
        (3, dict(s=1e10, pr=0.5, cum=0.5)),
        (3, dict(s=0, pr=1e-100, cum=0.5)),
        (4, dict(s=1, n=1, cum=1)),
        (4, dict(s=0, n=1, cum=0.5, pr=0.5)),
    ],
)
def test_invalid_unattainable_or_unidentified(which, kwargs):
    with pytest.raises(ValueError):
        cdf_binomial(which, **kwargs)


def test_backup_probability_outputs_are_not_assigned():
    cases = [c for c in FIXTURE["backup_reference"]["cases"] if c["input"][0] == 4]
    assert len(cases) == 25
    assert all(c["status"] == 0 and c["result"][-2:] == [0.123, 0.877] for c in cases)
    inputs = np.array([c["input"] for c in cases])
    result = cdf_binomial(4, cum=inputs[:, 1], ccum=inputs[:, 2], s=inputs[:, 3], n=inputs[:, 4])
    np.testing.assert_allclose(result.pr, inputs[:, 5], rtol=3e-12)
    np.testing.assert_allclose(result.cpr, inputs[:, 6], rtol=3e-12)


def test_backup_forward_probabilities_match_the_validated_definition():
    cases = [c for c in FIXTURE["backup_reference"]["cases"] if c["input"][0] == 1]
    inputs = np.array([c["input"] for c in cases])
    expected = np.array([c["result"][:2] for c in cases])
    result = cdf_binomial(s=inputs[:, 3], n=inputs[:, 4], pr=inputs[:, 5], cpr=inputs[:, 6])
    np.testing.assert_allclose(
        np.column_stack([result.cum, result.ccum]), expected, rtol=3e-12, atol=0
    )
