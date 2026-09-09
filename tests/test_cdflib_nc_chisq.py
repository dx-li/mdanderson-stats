"""Independent Poisson mixtures, shifted-normal identities and native evidence."""

import json
from decimal import Decimal, localcontext
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    ccum_nc_chisq,
    cdf_chisq,
    cdf_nc_chisq,
    cum_nc_chisq,
    inv_nc_chisq,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_nc_chisq.json").read_text())


def poisson_mixture(x, shape, nc):
    # Integer-shape gamma CDFs are finite Poisson sums. The outer Poisson
    # mixture is summed with 120-digit arithmetic and a negligible remainder.
    with localcontext() as ctx:
        ctx.prec = 120
        z, mean = Decimal(str(x)) / 2, Decimal(str(nc)) / 2
        weight, exponential = (-mean).exp(), (-z).exp()
        p = q = Decimal(0)
        for j in range(250):
            term = total = Decimal(1)
            for k in range(1, shape + j):
                term *= z / k
                total += term
            upper = exponential * total
            p += weight * (1 - upper)
            q += weight * upper
            weight *= mean / (j + 1)
        assert weight < Decimal("1e-100")
        return float(p), float(q)


@pytest.mark.parametrize("x", [0.2, 2, 10, 30])
@pytest.mark.parametrize("shape", [1, 2, 5, 10])
@pytest.mark.parametrize("nc", [0.5, 4, 20])
def test_independent_high_precision_poisson_mixture(x, shape, nc):
    p, q = poisson_mixture(x, shape, nc)
    r = cdf_nc_chisq(x=x, df=2 * shape, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
    assert float(inv_nc_chisq(p, 2 * shape, nc, ccum=q)) == pytest.approx(x, rel=3e-13)
    assert float(cdf_nc_chisq(3, cum=p, ccum=q, x=x, pnonc=nc).df) == pytest.approx(
        2 * shape, rel=3e-12
    )
    assert float(cdf_nc_chisq(4, cum=p, ccum=q, x=x, df=2 * shape).pnonc) == pytest.approx(
        nc, rel=3e-11
    )


@pytest.mark.parametrize("x", [0.1, 1, 10, 100])
@pytest.mark.parametrize("nc", [0.1, 4, 100])
def test_shifted_normal_square(x, nc):
    root, shift = sqrt(x), sqrt(nc)
    p = (erfc((shift - root) / sqrt(2)) - erfc((shift + root) / sqrt(2))) / 2
    q = (erfc((root - shift) / sqrt(2)) + erfc((root + shift) / sqrt(2))) / 2
    r = cdf_nc_chisq(x=x, df=1, pnonc=nc)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)


@pytest.mark.parametrize("case", FIXTURE["profiles"]["central_status_repaired"]["cases"])
def test_native_status_repaired_reference(case):
    which, p, q, x, df, nc = case["input"]
    if which == 1:
        r = cdf_nc_chisq(x=x, df=df, pnonc=nc)
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=0, atol=2e-6)
    else:
        kwargs = dict(cum=p, ccum=q, x=x, df=df, pnonc=nc)
        name = {2: "x", 3: "df", 4: "pnonc"}[which]
        truth = kwargs.pop(name)
        r = cdf_nc_chisq(which, **kwargs)
        # Native probabilities include truncation error, especially the
        # subtracted upper tail. Solve the actual request and verify it.
        forward = cdf_nc_chisq(x=r.x, df=r.df, pnonc=r.pnonc)
        actual = forward.cum if p <= q else forward.ccum
        assert float(actual) == pytest.approx(min(p, q), rel=3e-12)
        inaccurate = (x == 30 and df == 2 and nc == 0.5) or (
            which == 3 and x == 30 and df == 0.5 and nc == 4
        )
        if inaccurate:
            assert abs(float(getattr(r, name)) / truth - 1) > 3e-3
        else:
            assert float(getattr(r, name)) == pytest.approx(truth, rel=3e-3, abs=1e-4)


def test_original_status_failure_is_retained():
    original = FIXTURE["profiles"]["original"]
    assert not original["patches"]
    assert any(c["status"] == 10 for c in original["cases"])
    repaired = FIXTURE["profiles"]["central_status_repaired"]
    assert len(repaired["patches"]) == 1
    assert repaired["patches"][0]["file"] == "cdf_chisq_mod.f90"


@pytest.mark.parametrize("df,nc", [(0.001, 0), (0.001, 100), (5, 0), (1e10, 1e4), (5, 1e4)])
def test_domain_boundary_inversions_and_central_reduction(df, nc):
    x = df + nc
    r = cdf_nc_chisq(x=x, df=df, pnonc=nc)
    assert float(inv_nc_chisq(r.cum, df, nc, ccum=r.ccum)) == pytest.approx(x, rel=3e-12)
    assert float(cdf_nc_chisq(3, cum=r.cum, ccum=r.ccum, x=x, pnonc=nc).df) == pytest.approx(
        df, rel=3e-10
    )
    assert float(cdf_nc_chisq(4, cum=r.cum, ccum=r.ccum, x=x, df=df).pnonc) == pytest.approx(
        nc, rel=3e-10, abs=1e-12
    )
    if nc == 0:
        central = cdf_chisq(x=x, df=df)
        np.testing.assert_allclose([r.cum, r.ccum], [central.cum, central.ccum], rtol=3e-13)


@pytest.mark.parametrize("probability", [1e-300, 1e-100, 1e-20, 0.25])
def test_direct_extreme_tail_inversion(probability):
    for lower in (True, False):
        p, q = (probability, 1 - probability) if lower else (1 - probability, probability)
        if probability == 1e-300 and not lower:
            # Boost's upper-tail kernel underflows internally here. Do not
            # accept its finite but inconsistent inverse as a valid answer.
            try:
                extreme = inv_nc_chisq(p, 2, 4, ccum=q)
            except ArithmeticError as error:
                assert "forward verification" in str(error)
            else:
                # Permit future kernel repairs, but require independent accuracy.
                expected = poisson_mixture(float(extreme), 1, 4)[1]
                assert expected == pytest.approx(q, rel=1e-7, abs=0)
            continue
        x = inv_nc_chisq(p, 2, 4, ccum=q)
        r = cdf_nc_chisq(x=x, df=2, pnonc=4)
        assert float(r.cum if lower else r.ccum) == pytest.approx(probability, rel=1e-11, abs=0)


def test_broadcast_helpers_ownership_and_empty():
    x = np.array([[0.1], [1.0], [10.0]])
    nc = np.array([0.0, 0.5, 4.0])
    r = cdf_nc_chisq(x=x, df=2, pnonc=nc)
    np.testing.assert_array_equal(cum_nc_chisq(x, 2, nc), r.cum)
    np.testing.assert_array_equal(ccum_nc_chisq(x, 2, nc), r.ccum)
    np.testing.assert_allclose(inv_nc_chisq(r.cum, 2, nc, ccum=r.ccum), r.x, rtol=3e-13)
    x[:] = 5
    nc[:] = 5
    assert float(r.x[0, 0]) == 0.1 and float(r.pnonc[0, 0]) == 0
    for name in ("cum", "ccum", "x", "df", "pnonc"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert cdf_nc_chisq(x=[], df=1, pnonc=0).cum.size == 0
    assert inv_nc_chisq([], 1, 0).size == 0
    assert cdf_nc_chisq(3, cum=[], x=[], pnonc=1).df.size == 0
    assert cdf_nc_chisq(4, cum=[], x=[], df=1).pnonc.size == 0


def test_zero_and_large_x():
    r = cdf_nc_chisq(x=[0, 1e100], df=2, pnonc=4)
    np.testing.assert_array_equal(r.cum, [0, 1])
    np.testing.assert_array_equal(r.ccum, [1, 0])
    assert float(inv_nc_chisq(0, 2, 4)) == 0


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(x=1, df=1)),
        (1, dict(x=1, pnonc=1)),
        (1, dict(df=1, pnonc=1)),
        (1, dict(x=-1, df=1, pnonc=1)),
        (1, dict(x=1e101, df=1, pnonc=1)),
        (1, dict(x=1, df=0, pnonc=1)),
        (1, dict(x=1, df=1e11, pnonc=1)),
        (1, dict(x=1, df=1, pnonc=-1)),
        (1, dict(x=1, df=1, pnonc=1e5)),
        (1, dict(x=1, df=1, pnonc=np.nan)),
        (1, dict(x=1, df=1, pnonc=1, cum=0.5)),
        (2, dict(x=1, df=1, pnonc=1, cum=0.5)),
        (2, dict(df=1, pnonc=1, cum=1)),
        (3, dict(x=0, pnonc=1, cum=0.5)),
        (4, dict(x=0, df=1, cum=0.5)),
        (3, dict(x=1, pnonc=1, cum=0)),
        (4, dict(x=1, df=1, cum=1)),
        (3, dict(x=1, df=1, pnonc=1, cum=0.5)),
        (4, dict(x=1, df=1, pnonc=1, cum=0.5)),
        (3, dict(x=1, pnonc=100, cum=0.9)),
        (4, dict(x=1, df=100, cum=0.9)),
    ],
)
def test_invalid_unidentified_or_unattainable(which, kwargs):
    with pytest.raises(ValueError):
        cdf_nc_chisq(which, **kwargs)


@pytest.mark.parametrize("x,shape,nc,tail", [(0.2, 5, 20, 0), (30, 1, 0.5, 1)])
def test_native_early_stopping_and_subtracted_upper_tail_defects(x, shape, nc, tail):
    case = next(
        c
        for c in FIXTURE["profiles"]["central_status_repaired"]["cases"]
        if c["input"] == [1, 0.0, 0.0, x, 2 * shape, nc]
    )
    expected = poisson_mixture(x, shape, nc)[tail]
    assert abs(case["result"][tail] / expected - 1) > 0.1
    r = cdf_nc_chisq(x=x, df=2 * shape, pnonc=nc)
    assert float(r.cum if tail == 0 else r.ccum) == pytest.approx(expected, rel=3e-13, abs=0)


def test_unrepresentable_quantile_fails_forward_verification():
    with pytest.raises(ArithmeticError, match="forward verification"):
        inv_nc_chisq(1e-100, 0.001, 0)


@pytest.mark.parametrize("name,value", [("x", 1e100), ("df", 1e10), ("pnonc", 1e4)])
def test_strict_upper_input_bounds(name, value):
    kwargs = dict(x=1.0, df=1.0, pnonc=1.0)
    kwargs[name] = np.nextafter(value, np.inf)
    with pytest.raises(ValueError):
        cdf_nc_chisq(**kwargs)


def test_small_positive_noncentrality_is_not_replaced_with_zero():
    x, nc = 1.0, 1e-10
    r = cdf_nc_chisq(x=x, df=1, pnonc=nc)
    root = sqrt(nc)
    expected = (erfc((root - 1) / sqrt(2)) - erfc((root + 1) / sqrt(2))) / 2
    assert float(r.cum) == pytest.approx(expected, rel=3e-15)
    assert abs(float(r.cum) - float(cdf_chisq(x=x, df=1).cum)) > 1e-11
