"""Archived gamma references, independent identities and inversion contracts."""

import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import ccum_gamma, cdf_gamma, cum_gamma, inv_gamma

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_gamma.json").read_text())


def integer_gamma_tails(shape, z):
    with localcontext() as context:
        context.prec = 150
        zz = Decimal.from_float(float(z))
        term = total = Decimal(1)
        for k in range(1, shape):
            term *= zz / k
            total += term
        upper = (-zz).exp() * total
        return float(1 - upper), float(upper)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_gamma(case):
    which, p, q, x, shape, rate = case["input"]
    kwargs = dict(cum=p, ccum=q, x=x, shape=shape, rate=rate)
    for name in {1: ("cum", "ccum"), 2: ("x",), 3: ("shape",), 4: ("rate",)}[which]:
        kwargs.pop(name)
    result = cdf_gamma(which, **kwargs)
    if which == 1:
        assert case["status"] == 0
        if shape == 25 and x * rate <= 5:
            expected = integer_gamma_tails(25, x * rate)
            np.testing.assert_allclose([result.cum, result.ccum], expected, rtol=5e-13, atol=0)
            # Archived lower tails overstate the independently computed answer.
            assert 0.006 < case["result"][0] / expected[0] - 1 < 0.007
        else:
            np.testing.assert_allclose(
                [result.cum, result.ccum], case["result"][:2], rtol=2e-12, atol=0
            )
    else:
        # Generating parameters remain known even when the native inverse fails.
        np.testing.assert_allclose(
            [result.x, result.shape, result.rate], [x, shape, rate], rtol=2e-11
        )


@pytest.mark.parametrize(
    "shape,z,rate", list(itertools.product([1, 2, 5, 25], [0.01, 0.5, 3, 30], [0.1, 1, 7]))
)
def test_independent_integer_shape_poisson_identity(shape, z, rate):
    p, q = integer_gamma_tails(shape, z)
    x = z / rate
    result = cdf_gamma(x=x, shape=shape, rate=rate)
    np.testing.assert_allclose([result.cum, result.ccum], [p, q], rtol=5e-13, atol=0)
    assert float(inv_gamma(p, shape, rate, ccum=q)) == pytest.approx(x, rel=2e-13)
    assert float(cdf_gamma(3, x=x, rate=rate, cum=p, ccum=q).shape) == pytest.approx(
        shape, rel=2e-12
    )
    assert float(cdf_gamma(4, x=x, shape=shape, cum=p, ccum=q).rate) == pytest.approx(
        rate, rel=2e-13
    )


@pytest.mark.parametrize("probability", [1e-300, 1e-100, 1e-20, 0.01, 0.5])
def test_extreme_exponential_quantiles_and_rates(probability):
    for p, q, z in (
        (probability, 1 - probability, -np.log1p(-probability)),
        (1 - probability, probability, -np.log(probability)),
    ):
        x = float(inv_gamma(p, 1, 3, ccum=q))
        assert x == pytest.approx(z / 3, rel=2e-13, abs=0)
        r = cdf_gamma(x=x, shape=1, rate=3)
        np.testing.assert_allclose([r.cum, r.ccum], [p, q], rtol=3e-13, atol=0)
        assert float(cdf_gamma(4, x=x, shape=1, cum=p, ccum=q).rate) == pytest.approx(3, rel=2e-13)
        assert float(cdf_gamma(3, x=x, rate=3, cum=p, ccum=q).shape) == pytest.approx(1, rel=2e-12)


def test_broadcast_helpers_and_immutable_ownership():
    x = np.array([[0.2], [1], [5]])
    shape = np.array([0.2, 1, 3])
    r = cdf_gamma(x=x, shape=shape)
    assert r.cum.shape == (3, 3)
    np.testing.assert_array_equal(cum_gamma(x, shape), r.cum)
    np.testing.assert_array_equal(ccum_gamma(x, shape), r.ccum)
    np.testing.assert_array_equal(r.cum + r.ccum, np.ones((3, 3)))
    np.testing.assert_allclose(inv_gamma(r.cum, shape, ccum=r.ccum), r.x, rtol=1e-13)
    x[:] = 2
    shape[:] = 2
    assert float(r.x[0, 0]) == 0.2 and float(r.shape[0, 0]) == 0.2
    for name in ("cum", "ccum", "x", "shape", "rate"):
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for which in (1, 2, 3, 4):
        kwargs = dict(x=[], shape=[], rate=[], cum=[], ccum=[])
        for name in {1: ("cum", "ccum"), 2: ("x",), 3: ("shape",), 4: ("rate",)}[which]:
            kwargs.pop(name)
        assert cdf_gamma(which, **kwargs).cum.shape == (0,)


@pytest.mark.parametrize("shape", [1e-10, 0.1, 1, 1e5, 1e100])
def test_zero_coordinate_and_quantile(shape):
    r = cdf_gamma(x=0, shape=shape)
    assert float(r.cum) == 0 and float(r.ccum) == 1
    assert float(inv_gamma(0, shape)) == 0


@pytest.mark.parametrize("rate,z", list(itertools.product([1e-10, 1e100], [0.1, 0.5, 1, 2, 5])))
def test_rate_boundaries(rate, z):
    r = cdf_gamma(x=z / rate, shape=1, rate=rate)
    assert float(cdf_gamma(4, x=r.x, shape=1, cum=r.cum, ccum=r.ccum).rate) == pytest.approx(
        rate, rel=2e-14
    )


@pytest.mark.parametrize("shape", [1e-10, 1e-5, 1e2, 1e5, 1e10])
def test_shape_round_trip_across_magnitudes(shape):
    r = cdf_gamma(x=shape, shape=shape)
    assert float(cdf_gamma(3, x=r.x, cum=r.cum, ccum=r.ccum).shape) == pytest.approx(
        shape, rel=1e-13
    )


def test_original_rate_iteration_status_bug_and_shape_initial_root_failure():
    rate_failures = [c for c in FIXTURE["cases"] if c["input"][0] == 4 and c["status"] == 10]
    shape_failures = [c for c in FIXTURE["cases"] if c["input"][0] == 3 and c["status"] == -50]
    assert len(rate_failures) == 36
    assert len(shape_failures) == 9
    assert all(c["input"][4] == 5 for c in shape_failures)
    for case in rate_failures + shape_failures:
        test_native_gamma(case)


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, {"shape": 1}),
        (1, {"x": 1}),
        (1, {"x": -1, "shape": 1}),
        (1, {"x": 1e101, "shape": 1}),
        (1, {"x": 1, "shape": 0}),
        (1, {"x": 1, "shape": 1e101}),
        (1, {"x": 1, "shape": 1, "rate": 0}),
        (1, {"x": 1, "shape": 1, "rate": np.inf}),
        (1, {"x": np.nan, "shape": 1}),
        (1, {"x": 1, "shape": 1, "cum": 0.5}),
        (2, {"shape": 1}),
        (2, {"shape": 1, "cum": 1}),
        (2, {"shape": 1, "cum": 0.5, "ccum": 0.6}),
        (2, {"shape": 1, "cum": -0.1}),
        (2, {"shape": 1, "cum": 0.5, "x": 1}),
        (3, {"x": 0, "cum": 0.5}),
        (3, {"x": 1, "cum": 0}),
        (3, {"x": 1e100, "rate": 1e100, "cum": 0.5}),
        (4, {"x": 0, "shape": 1, "cum": 0.5}),
        (4, {"x": 1, "shape": 1, "cum": 0}),
        (4, {"x": 1e-300, "shape": 1, "cum": 0.5}),
        (4, {"x": 1e100, "shape": 1, "cum": 0.5}),
    ],
)
def test_invalid_inputs_and_unattainable_solutions(which, kwargs):
    with pytest.raises(ValueError):
        cdf_gamma(which, **kwargs)


def test_strict_input_bounds():
    for name in ("shape", "rate"):
        for value in (np.nextafter(1e-10, 0), np.nextafter(1e100, np.inf)):
            kwargs = dict(x=1, shape=1, rate=1)
            kwargs[name] = value
            with pytest.raises(ValueError):
                cdf_gamma(**kwargs)
