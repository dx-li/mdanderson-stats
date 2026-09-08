"""Native pilot MSE diagnostics and independent convolution properties."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import muhaz_mse

CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-mse.json").read_text())["cases"]
KERNELS = ["rectangle", "epanechnikov", "biquadratic", "triquadratic"]
BOUNDARIES = ["none", "left", "both"]


@pytest.mark.parametrize("case", CASES)
def test_original_msemse(case):
    r = muhaz_mse(
        case["times"],
        case["delta"],
        bandwidths=case["bandwidths"],
        grid=case["grid"],
        pilot_bandwidth=case["pilot_bandwidth"],
        kernel=KERNELS[case["kernel"]],
        boundary=BOUNDARIES[case["boundary"]],
        legacy=True,
    )
    assert_allclose(
        np.stack([r.bias, r.variance, r.mse], axis=-1), case["diagnostics"], rtol=2e-10, atol=2e-13
    )
    assert_allclose(r.mse, r.bias**2 + r.variance)
    assert np.all((r.refinements >= 2) & (r.refinements <= 6))


def test_constant_rectangle_pilot_convolution_against_exact_integral():
    # Pilot hazard is 1/(2*10)=.05 throughout this convolution interval.
    # Survival factor is 1 below the event and 1/2 above it. Thus the
    # variance integral is .05/4*(1+2)=.0375; divide by N*b=.2.
    r = muhaz_mse(
        [1],
        bandwidths=0.2,
        pilot_bandwidth=10,
        grid=[1],
        bounds=(0, 2),
        kernel="rectangle",
        boundary="none",
        rtol=0,
        max_refinements=12,
    )
    assert_allclose(r.pilot_hazard, [0.05])
    assert_allclose(r.bias, 0, atol=1e-16)
    assert_allclose(r.variance, [[0.1875]], atol=4e-5)
    assert not r.converged[0, 0]
    assert r.refinements[0, 0] == 12


def test_zero_event_integrals_converge_and_one_level_is_reported_unconverged():
    kw = dict(times=[1, 2], delta=[0, 0], bandwidths=[0.2, 0.4], pilot_bandwidth=0.3)
    r = muhaz_mse(**kw)
    assert_array_equal(r.mse, 0)
    assert_array_equal(r.converged, True)
    assert_array_equal(r.refinements, 2)
    limited = muhaz_mse(**kw, max_refinements=1)
    assert_array_equal(limited.converged, False)
    assert_array_equal(limited.refinements, 1)


def test_permutation_time_scaling_and_candidate_order():
    t = np.array([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3])
    d = np.array([1, 0, 1, 1, 0, 1, 0])
    first = muhaz_mse(t, d, bandwidths=[0.3, 0.8], pilot_bandwidth=0.65, grid=[0.1, 1.1, 2.9])
    second = muhaz_mse(
        t[::-1] * 2, d[::-1], bandwidths=[1.6, 0.6], pilot_bandwidth=1.3, grid=[0.2, 2.2, 5.8]
    )
    assert_allclose(second.bias, first.bias[::-1] / 2, atol=1e-15)
    assert_allclose(second.variance, first.variance[::-1] / 4, atol=1e-15)
    assert_allclose(second.mse, first.mse[::-1] / 4, atol=1e-15)
    with pytest.raises(ValueError):
        first.mse[0, 0] = 1


def test_empty_grid_retains_bandwidth_axis():
    r = muhaz_mse([1, 2], bandwidths=[0.2, 0.3], pilot_bandwidth=0.4, grid=[])
    assert r.mse.shape == (2, 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"bandwidths": []},
        {"bandwidths": [0]},
        {"bandwidths": [[1]]},
        {"pilot_bandwidth": 0},
        {"rtol": -1},
        {"rtol": np.inf},
        {"max_refinements": 0},
        {"max_refinements": 17},
        {"max_refinements": True},
        {"times": []},
        {"grid": [-1]},
    ],
)
def test_invalid_inputs(overrides):
    args = dict(times=[1, 2], bandwidths=[0.2], pilot_bandwidth=0.3)
    args.update(overrides)
    with pytest.raises(ValueError):
        muhaz_mse(**args)
