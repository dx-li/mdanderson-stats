"""Source parity and independently integrated corrected piecewise hazard law."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    calibrate_prt_hazard,
    prt_arrivals_from_uniforms,
    prt_hazard_delays,
)

ROOT = Path(__file__).parent / "fixtures"
CASES = json.loads((ROOT / "prt-desktop-hazard.json").read_text())["cases"]


@pytest.mark.parametrize("c", CASES)
def test_legacy_hazard_matches_native_calibration_and_delay(c):
    model = calibrate_prt_hazard(
        c["probability"], c["window"], c["alpha"], c["beta"], profile="legacy_1_0"
    )
    assert model.iterations == c["iterations"]
    assert model.scale == pytest.approx(c["scale"], abs=1e-8)
    np.testing.assert_allclose(model.rates, c["rates"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(model.cdf, c["cdf"], rtol=0, atol=1e-12)
    result = prt_hazard_delays(model, [s["first"] for s in c["samples"]])
    np.testing.assert_allclose(result, [s["delay"] for s in c["samples"]], rtol=1e-11, atol=1e-8)
    assert not result.flags.writeable


@pytest.mark.parametrize("c", CASES)
def test_corrected_inverse_matches_independent_integrated_hazard(c):
    model = calibrate_prt_hazard(c["probability"], c["window"], c["alpha"], c["beta"])
    assert model.achieved_probability == pytest.approx(c["probability"], abs=1e-14)
    uniforms = np.r_[0.0, (np.arange(1000) + 0.5) / 1000, np.nextafter(1.0, 0.0)]
    delays = prt_hazard_delays(model, uniforms)
    for u, t in zip(uniforms, delays, strict=True):
        if u >= c["probability"]:
            assert np.isinf(t)
        else:
            integral = sum(
                max(0.0, min(t, right) - left) * rate
                for left, right, rate in zip(
                    model.grid[:-2], model.grid[1:-1], model.rates[:98], strict=True
                )
            )
            assert -np.expm1(-integral) == pytest.approx(u, abs=2e-14)
            assert 0 <= t < c["window"]
    assert np.isfinite(delays[1:-1]).sum() == round(c["probability"] * 1000)


def test_uniform_hazard_has_exponential_event_law_until_window():
    model = calibrate_prt_hazard(0.3, 90, 1, 1)
    u = np.array([0.0, 0.01, 0.1, 0.29, 0.31])
    expected = -np.log1p(-u) / (-np.log(0.7) / 90)
    expected[u >= 0.3] = np.inf
    np.testing.assert_allclose(prt_hazard_delays(model, u), expected, rtol=2e-15, atol=1e-14)
    assert prt_hazard_delays(model, 0.1).shape == ()


@pytest.mark.parametrize("c", json.loads((ROOT / "prt-desktop-arrivals.json").read_text())["cases"])
def test_finite_uniform_arrivals_match_native(c):
    np.testing.assert_allclose(
        prt_arrivals_from_uniforms(c["mean"], c["uniforms"]), c["arrivals"], rtol=2e-16, atol=1e-11
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probability": 0},
        {"probability": 1},
        {"window": 0},
        {"alpha": 0},
        {"beta": -1},
        {"profile": "unknown"},
        {"alpha": 10001},
        {"window": 1e7},
    ],
)
def test_calibration_rejects_invalid_inputs(kwargs):
    args = dict(probability=0.3, window=90, alpha=5, beta=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        calibrate_prt_hazard(**args)


@pytest.mark.parametrize("u", [-0.1, 1.0, float("nan"), float("inf")])
def test_inverse_rejects_invalid_uniforms(u):
    with pytest.raises(ValueError):
        prt_hazard_delays(calibrate_prt_hazard(0.3, 90, 5, 1), u)


def test_result_arrays_are_immutable_and_metadata_checked():
    model = calibrate_prt_hazard(0.3, 90, 5, 1)
    for a in (model.grid, model.rates, model.cdf):
        assert not a.flags.writeable
    with pytest.raises(ValueError):
        replace(model, cdf=np.zeros(100))
    with pytest.raises(ValueError):
        replace(model, profile="unknown")
    with pytest.raises(ValueError):
        replace(model, requested_probability=0.8)


def test_arrival_shape_and_overflow_guards():
    with pytest.raises(ValueError):
        prt_arrivals_from_uniforms(1, [0.5] * 799)
    with pytest.raises(ArithmeticError):
        prt_arrivals_from_uniforms(1e308, [0.5] * 800)
