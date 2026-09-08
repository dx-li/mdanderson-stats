"""Local selection, bandwidth smoothing and variable-bandwidth hazard checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import muhaz_fixed, muhaz_global, muhaz_local
from mdanderson_stats.muhaz_local import _smooth_bandwidth

CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-local.json").read_text())["cases"]
KERNELS = ["rectangle", "epanechnikov", "biquadratic", "triquadratic"]
BOUNDARIES = ["none", "left", "both"]


@pytest.mark.parametrize("case", CASES)
def test_original_local_fit(case):
    r = muhaz_local(
        case["times"],
        case["delta"],
        bandwidths=case["bandwidths"],
        pilot_bandwidth=0.65,
        smoothing_bandwidth=case["smoothing"],
        bounds=(0, 3),
        n_min_grid=9,
        n_est_grid=25,
        kernel=KERNELS[case["kernel"]],
        boundary=BOUNDARIES[case["boundary"]],
        legacy=True,
    )
    assert_allclose(r.hazard, case["hazard"], rtol=3e-10, atol=3e-12)
    assert_allclose(r.bandwidth, case["bandwidth"], rtol=3e-12)
    if case["local_bandwidth"] is None:
        assert r.local_bandwidth is r.diagnostics is r.score is r.minimum_mse is None
        assert r.smoothing_bandwidth is None
    else:
        assert_allclose(r.local_bandwidth, case["local_bandwidth"])
        assert_allclose(r.minimum_mse, case["minimum_mse"], rtol=3e-10, atol=3e-12)
        assert_allclose(r.score, case["score"], rtol=3e-10)
        available = r.minimum_mse < 1e30
        assert_allclose(
            r.selected_bias[available], np.array(case["bias"])[available], rtol=3e-10, atol=3e-12
        )
        assert_allclose(
            r.selected_variance[available],
            np.array(case["variance"])[available],
            rtol=3e-10,
            atol=3e-12,
        )
        assert np.isnan(r.selected_bias[~available]).all()
        assert np.isnan(r.selected_variance[~available]).all()


def test_each_point_minimizes_mse_and_hazard_uses_its_smoothed_bandwidth():
    t = [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3]
    d = [1, 0, 1, 1, 0, 1, 0]
    r = muhaz_local(t, d, bandwidths=[0.35, 0.8, 2], pilot_bandwidth=0.65, bounds=(0, 3))
    assert_array_equal(r.selected_index, r.diagnostics.mse.argmin(axis=0))
    assert_allclose(r.minimum_mse, r.diagnostics.mse.min(axis=0))
    assert r.score == r.minimum_mse.sum()
    for i in [0, 15, 50, 75, 100]:
        fixed = muhaz_fixed(t, d, bandwidth=r.bandwidth[i], grid=[r.time[i]])
        assert_allclose(r.hazard[i], fixed.hazard[0])
    assert r.smoothing_bandwidth == 5 * 0.65
    for array in [
        r.time,
        r.hazard,
        r.bandwidth,
        r.local_bandwidth,
        r.selected_index,
        r.minimum_mse,
        r.selected_bias,
        r.selected_variance,
    ]:
        assert not array.flags.writeable


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_smoothing_preserves_constant_bandwidth(kernel, boundary):
    grid = np.linspace(0, 3, 17)
    r = _smooth_bandwidth(
        grid, np.full(17, 0.7), np.linspace(0, 3, 31), 1.1, (0, 3), kernel, boundary, False
    )
    assert_allclose(r, 0.7, rtol=2e-14)


def test_rectangle_interior_smoothing_is_arithmetic_mean():
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    bw = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    result = _smooth_bandwidth(
        grid, bw, np.array([1.5, 2.5]), 1.0, (0, 4), "rectangle", "none", False
    )
    assert_allclose(result, [3, 6])


def test_left_only_smoother_omits_right_correction_unless_legacy():
    grid = np.linspace(0, 3, 9)
    bw = 0.5 + grid**2
    args = (grid, bw, np.array([2.8]), 0.7, (0, 3), "epanechnikov")
    left = _smooth_bandwidth(*args, "left", False)
    none = _smooth_bandwidth(*args, "none", False)
    both = _smooth_bandwidth(*args, "both", False)
    old = _smooth_bandwidth(*args, "left", True)
    assert_allclose(left, none)
    assert_allclose(old, both)
    assert not np.allclose(left, old)


def test_zero_score_selection_and_unavailable_legacy_diagnostics():
    args = dict(
        times=[1, 2], delta=[0, 0], bandwidths=[0.3, 0.8], pilot_bandwidth=0.5, bounds=(0, 2)
    )
    r = muhaz_local(**args)
    assert_allclose(r.bandwidth, 0.3)
    assert_array_equal(r.minimum_mse, 0)
    old = muhaz_local(**args, legacy=True)
    assert_allclose(old.bandwidth, 0.8)
    assert_array_equal(old.minimum_mse, 1e30)
    assert np.isnan(old.selected_bias).all()
    assert np.isnan(old.selected_variance).all()
    assert_array_equal(old.hazard, 0)


def test_shared_defaults_subset_and_single_candidate_bypass():
    args = dict(
        times=[1, 2, np.nan],
        delta=[1, 0, np.nan],
        subset=[True, True, False],
        bounds=(0, 4),
        bandwidths=0.5,
    )
    local, glob = muhaz_local(**args), muhaz_global(**args)
    assert_array_equal(local.hazard, glob.hazard)
    assert local.bounds == glob.curve.bounds
    assert local.n_observations == 2 and local.n_events == 1
    assert local.diagnostics is local.score is None
    t = np.arange(1, 22)
    r = muhaz_local(t, n_min_grid=7, n_est_grid=9)
    g = muhaz_global(t, n_min_grid=7, n_est_grid=9)
    assert r.bounds == g.curve.bounds
    assert_array_equal(r.diagnostics.bandwidths, g.bandwidths)


def test_scaling_preserves_bandwidth_selection():
    args = dict(
        bandwidths=np.array([0.35, 0.8, 2]),
        pilot_bandwidth=0.65,
        smoothing_bandwidth=0.8,
        bounds=(0, 3),
        n_min_grid=9,
        n_est_grid=13,
    )
    t = np.array([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3])
    a = muhaz_local(t, **args)
    args.update(
        bandwidths=args["bandwidths"] * 2,
        pilot_bandwidth=1.3,
        smoothing_bandwidth=1.6,
        bounds=(0, 6),
    )
    b = muhaz_local(t * 2, **args)
    assert_array_equal(a.selected_index, b.selected_index)
    assert_allclose(a.bandwidth * 2, b.bandwidth)
    assert_allclose(a.hazard / 2, b.hazard)
    assert_allclose(a.minimum_mse / 4, b.minimum_mse)


@pytest.mark.parametrize("smoothing", [0, -1, np.nan, np.inf])
def test_invalid_smoothing(smoothing):
    with pytest.raises(ValueError):
        muhaz_local([1, 2], bandwidths=0.5, bounds=(0, 2), smoothing_bandwidth=smoothing)


def test_uncovered_smoothing_point_raises_instead_of_fabricating_bandwidth():
    with pytest.raises(RuntimeError, match="undefined"):
        muhaz_local(
            [1, 2],
            bandwidths=[0.2, 0.8],
            pilot_bandwidth=0.4,
            smoothing_bandwidth=0.01,
            bounds=(0, 2),
            n_min_grid=2,
            n_est_grid=3,
        )


def test_negative_boundary_weight_can_make_smoothing_invalid():
    with pytest.raises(RuntimeError, match="finite and positive"):
        _smooth_bandwidth(
            np.linspace(0, 1, 5),
            np.array([1.0, 1.0, 1.0, 100.0, 1.0]),
            np.array([0.0]),
            1.0,
            (0, 1),
            "epanechnikov",
            "both",
            False,
        )


def test_variable_bandwidth_evaluation_crosses_chunk_boundaries():
    t = np.linspace(0.1, 3, 1000)
    r = muhaz_local(
        t, bandwidths=[0.3, 0.6], pilot_bandwidth=0.5, bounds=(0, 3), n_min_grid=9, n_est_grid=700
    )
    for i in [0, 261, 262, 523, 524, 699]:
        fixed = muhaz_fixed(t, bandwidth=r.bandwidth[i], grid=[r.time[i]], bounds=(0, 3))
        assert_allclose(r.hazard[i], fixed.hazard[0], rtol=1e-13)
