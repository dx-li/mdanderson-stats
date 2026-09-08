"""Native Poisson references and independent discrete/continuous identities."""

import json
from decimal import Decimal, localcontext
from math import erfc, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_poisson, cdf_poisson, cum_poisson, inv_poisson

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_poisson.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_poisson(case):
    which, p, q, s, mean = case["input"]
    kwargs = dict(cum=p, ccum=q, s=s, mean=mean)
    for name in {1: ("cum", "ccum"), 2: ("s",), 3: ("mean",)}[which]:
        kwargs.pop(name)
    r = cdf_poisson(which, **kwargs)
    if which == 1:
        assert case["status"] == 0
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=2e-12, atol=0)
    else:
        np.testing.assert_allclose([r.s, r.mean], [s, mean], rtol=2e-11, atol=1e-13)


@pytest.mark.parametrize("s", [0, 1, 4, 10])
@pytest.mark.parametrize("mean", [1e-10, 0.1, 1, 5, 30])
def test_independent_poisson_sum(s, mean):
    with localcontext() as context:
        context.prec = 150
        mu = Decimal.from_float(mean)
        total = term = Decimal(1)
        for k in range(1, s + 1):
            term *= mu / k
            total += term
        lower = (-mu).exp() * total
        p, q = float(lower), float(1 - lower)
    r = cdf_poisson(s=s, mean=mean)
    np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=2e-13, atol=0)
    assert float(cdf_poisson(3, s=s, cum=p, ccum=q).mean) == pytest.approx(mean, rel=2e-13)
    assert float(inv_poisson(p, mean, ccum=q)) == pytest.approx(s, rel=2e-12, abs=1e-13)


@pytest.mark.parametrize("mean", [0.1, 1, 5, 30])
def test_fractional_count_half_integer_gamma_identity(mean):
    p = erfc(sqrt(mean)) + 2 * sqrt(mean / pi) * exp(-mean)
    r = cdf_poisson(s=0.5, mean=mean)
    assert float(r.cum) == pytest.approx(p, rel=2e-13)
    assert float(inv_poisson(p, mean)) == pytest.approx(0.5, rel=2e-11)
    assert float(cdf_poisson(3, s=0.5, cum=p).mean) == pytest.approx(mean, rel=2e-12)


def test_broadcast_helpers_ownership_empty_and_extreme_mean_inverse():
    s = np.array([[0.0], [0.5], [3]])
    mean = np.array([0.2, 1, 5])
    r = cdf_poisson(s=s, mean=mean)
    assert r.cum.shape == (3, 3)
    np.testing.assert_array_equal(cum_poisson(s, mean), r.cum)
    np.testing.assert_array_equal(ccum_poisson(s, mean), r.ccum)
    np.testing.assert_allclose(inv_poisson(r.cum, mean, ccum=r.ccum), r.s, atol=1e-13)
    s[:] = 0
    mean[:] = 0
    assert float(r.s[1, 0]) == 0.5 and float(r.mean[0, 0]) == 0.2
    for name in ("s", "mean", "cum", "ccum"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which in (1, 2, 3):
        kwargs = dict(s=[], mean=[], cum=[], ccum=[])
        for name in {1: ("cum", "ccum"), 2: ("s",), 3: ("mean",)}[which]:
            kwargs.pop(name)
        assert cdf_poisson(which, **kwargs).cum.size == 0
    assert float(cdf_poisson(3, s=0, cum=1e-300).mean) == pytest.approx(-np.log(1e-300))


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (4, {}),
        (1.0, {}),
        (1, {"s": 0}),
        (1, {"mean": 1}),
        (1, {"s": -1, "mean": 1}),
        (1, {"s": 1e101, "mean": 1}),
        (1, {"s": 0, "mean": 0}),
        (1, {"s": 0, "mean": np.inf}),
        (1, {"s": 0, "mean": 1, "cum": 0.5}),
        (2, {"cum": 0, "mean": 1}),
        (2, {"cum": 1, "mean": 1}),
        (2, {"cum": 0.1, "mean": 1}),
        (2, {"cum": 0.5, "mean": 1, "s": 1}),
        (3, {"s": 0, "cum": 1}),
        (3, {"s": 0, "cum": 0}),
        (3, {"s": 0, "ccum": 1e-20}),
        (3, {"s": 0, "cum": 0.5, "mean": 1}),
    ],
)
def test_invalid_unattainable_and_output_supplied(which, kwargs):
    with pytest.raises(ValueError):
        cdf_poisson(which, **kwargs)


def test_strict_mean_bounds():
    for mean in (np.nextafter(1e-10, 0), np.nextafter(1e100, np.inf)):
        with pytest.raises(ValueError):
            cdf_poisson(s=0, mean=mean)


@pytest.mark.parametrize("case", FIXTURE["unattainable_count_cases"])
def test_native_negative_count_is_silently_clamped(case):
    assert case["status"] == 0 and case["result"][2] == 0
    which, p, q, s, mean = case["input"]
    assert p < exp(-mean)
    with pytest.raises(ValueError):
        cdf_poisson(which, cum=p, ccum=q, mean=mean)
