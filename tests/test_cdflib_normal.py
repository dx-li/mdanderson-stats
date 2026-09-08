"""Native references and independent normal location-scale identities."""

import itertools
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_normal, cdf_normal, cum_normal, inv_normal

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_normal.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_normal(case):
    which, p, q, x, mean, sd = case["input"]
    kwargs = dict(cum=p, ccum=q, x=x, mean=mean, sd=sd)
    for name in {1: ("cum", "ccum"), 2: ("x",), 3: ("mean",), 4: ("sd",)}[which]:
        kwargs.pop(name)
    assert case["status"] == 0
    result = cdf_normal(which, **kwargs)
    actual = [result.cum, result.ccum, result.x, result.mean, result.sd]
    np.testing.assert_allclose(actual, case["result"], rtol=3e-12, atol=1e-13)


@pytest.mark.parametrize(
    "z,mean,sd",
    list(
        itertools.product(
            [-8.0, -2.0, -0.5, 0.0, 0.7, 3.0, 8.0], [-2.0, 0.0, 10.0], [0.2, 1.0, 5.0]
        )
    ),
)
def test_independent_erfc_and_location_scale_inversions(z, mean, sd):
    p, q = 0.5 * erfc(-z / sqrt(2)), 0.5 * erfc(z / sqrt(2))
    x = mean + sd * z
    result = cdf_normal(x=x, mean=mean, sd=sd)
    np.testing.assert_allclose([result.cum, result.ccum], [p, q], rtol=2e-13, atol=0)
    assert float(cdf_normal(2, cum=p, ccum=q, mean=mean, sd=sd).x) == pytest.approx(
        x, rel=1e-12, abs=1e-13
    )
    assert float(cdf_normal(3, cum=p, ccum=q, x=x, sd=sd).mean) == pytest.approx(
        mean, rel=1e-12, abs=1e-13
    )
    if z:
        assert float(cdf_normal(4, cum=p, ccum=q, x=x, mean=mean).sd) == pytest.approx(
            sd, rel=1e-12
        )


def test_tiny_complements_and_reflection():
    q = 0.5 * erfc(37 / sqrt(2))
    assert float(inv_normal(ccum=q)) == pytest.approx(37, rel=1e-14)
    assert float(inv_normal(q)) == pytest.approx(-37, rel=1e-14)
    result = cdf_normal(x=37)
    assert float(result.cum) == 1
    assert float(result.ccum) == pytest.approx(q, rel=2e-12, abs=0)
    assert float(cdf_normal(4, x=74, ccum=q).sd) == pytest.approx(2)
    assert float(cdf_normal(3, x=74, ccum=q, sd=2).mean) == pytest.approx(0, abs=1e-12)


def test_defaults_broadcast_helpers_and_immutable_ownership():
    x = np.array([[-2.0], [0.0], [3.0]])
    mean = np.array([0.0, 1.0])
    r = cdf_normal(x=x, mean=mean)
    assert r.cum.shape == (3, 2)
    np.testing.assert_array_equal(cum_normal(x, mean), r.cum)
    np.testing.assert_array_equal(ccum_normal(x, mean), r.ccum)
    np.testing.assert_array_equal(r.cum + r.ccum, np.ones((3, 2)))
    np.testing.assert_allclose(inv_normal(r.cum, mean, ccum=r.ccum), r.x, atol=1e-14)
    np.testing.assert_allclose(cdf_normal(3, x=x, cum=r.cum, ccum=r.ccum).mean, r.mean, atol=1e-14)
    x[:] = 1
    mean[:] = 2
    assert float(r.x[0, 0]) == -2 and float(r.mean[0, 0]) == 0
    for name in ("cum", "ccum", "x", "mean", "sd"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    assert float(inv_normal(0.5)) == 0
    assert float(cdf_normal(3, x=5, cum=0.5).mean) == 5
    assert cdf_normal(x=[]).cum.shape == (0,)


@pytest.mark.parametrize("case", FIXTURE["invalid_sd_cases"])
def test_source_success_can_return_invalid_sd(case):
    which, p, q, x, mean, sd = case["input"]
    assert case["status"] == 0
    native_sd = case["result"][-1]
    assert isinstance(native_sd, str) or native_sd < 0
    with pytest.raises(ValueError):
        cdf_normal(which, cum=p, ccum=q, x=x, mean=mean)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1, {}),
        (1, {"x": 0, "sd": 0}),
        (1, {"x": 0, "sd": 1e-11}),
        (1, {"x": 0, "mean": float("inf")}),
        (1, {"x": 1e101}),
        (1, {"x": 0, "cum": 0.5}),
        (2, {"cum": 0}),
        (2, {"ccum": 0}),
        (2, {"cum": 0.4, "ccum": 0.5}),
        (2, {"cum": 0.5, "x": 0}),
        (3, {"cum": 0.5, "x": 0, "mean": 0}),
        (4, {"cum": 0.5, "x": 0, "sd": 1}),
        (4, {"cum": 0.9, "x": 0}),
        (2, {"cum": 0.99, "mean": 1e100, "sd": 1e100}),
        (3, {"cum": 0.01, "x": 1e100, "sd": 1e100}),
        (4, {"cum": 0.8, "x": 1e100, "mean": -1e100}),
    ],
)
def test_invalid_missing_and_out_of_domain_solutions(which, kwargs):
    with pytest.raises(ValueError):
        cdf_normal(which, **kwargs)


def test_finite_domain_boundaries_and_underflow():
    result = cdf_normal(x=[-1e100, 0, 1e100], sd=1e-10)
    np.testing.assert_array_equal(result.cum, [0, 0.5, 1])
    np.testing.assert_array_equal(result.ccum, [1, 0.5, 0])
    assert float(cdf_normal(2, cum=0.5, mean=1e100, sd=1e100).x) == 1e100
    assert float(cdf_normal(3, cum=0.5, x=-1e100, sd=1e-10).mean) == -1e100


@pytest.mark.parametrize(
    "sd,z",
    [(1e-10, z) for z in (-5, -2, -1, -0.5, 0.5, 1, 2, 5)]
    + [(1e100, z) for z in (-1, -0.5, 0.5, 1)],
)
def test_sd_boundary_round_trips_allow_only_arithmetic_roundoff(sd, z):
    x = sd * z
    r = cdf_normal(x=x, sd=sd)
    answer = cdf_normal(4, x=x, cum=r.cum, ccum=r.ccum).sd
    assert 1e-10 <= float(answer) <= 1e100
    assert float(answer) == pytest.approx(sd, rel=2e-14)


def test_input_bounds_remain_strict():
    with pytest.raises(ValueError, match="sd"):
        cdf_normal(x=0, sd=np.nextafter(1e-10, 0))
    with pytest.raises(ValueError, match="sd"):
        cdf_normal(x=0, sd=np.nextafter(1e100, np.inf))
