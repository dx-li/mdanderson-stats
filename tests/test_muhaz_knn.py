"""Native complete neighbor fits and independent selection contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import muhaz_fixed, muhaz_knn, muhaz_mse

CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-knn.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_knnhad(case):
    args = dict(
        times=case["times"],
        delta=case["delta"],
        neighbors=case["neighbors"],
        method="failures" if case["method"] == 1 else "survival",
        pilot_bandwidth=0.65,
        smoothing_bandwidth=case["smoothing"],
        bounds=(0, 3),
        n_min_grid=9,
        n_est_grid=25,
        kernel=["rectangle", "epanechnikov", "biquadratic", "triquadratic"][case["kernel"]],
        boundary=["none", "left", "both"][case["boundary"]],
        legacy=True,
    )
    if min(case["bandwidth"]) <= 0:
        with pytest.raises(RuntimeError, match="finite and positive"):
            muhaz_knn(**args)
        return
    r = muhaz_knn(**args)
    assert r.neighbors == case["selected_neighbors"]
    assert_allclose(
        r.neighbor_bandwidths.bandwidth[r.selected_index], case["local_bandwidth"], rtol=3e-12
    )
    assert_allclose(r.bandwidth, case["bandwidth"], rtol=3e-12)
    assert_allclose(r.hazard, case["hazard"], rtol=4e-10, atol=3e-12)
    if case["scores"] is None:
        assert r.score is r.scores is r.diagnostics is None
    else:
        assert_allclose(r.scores, case["scores"], rtol=4e-10, atol=3e-12)
        assert r.score == r.scores.min()


def test_varying_bandwidth_mse_agrees_with_individual_constant_evaluations():
    t = [0.2, 0.7, 1.3, 2, 3]
    z = [0.1, 1.1, 2.9]
    bw = np.array([[0.3, 0.4, 0.5], [0.7, 0.8, 0.9]])
    r = muhaz_mse(t, bandwidths=bw, pilot_bandwidth=0.6, grid=z)
    for row in range(2):
        for col in range(3):
            one = muhaz_mse(t, bandwidths=bw[row, col], pilot_bandwidth=0.6, grid=[z[col]])
            assert_allclose(r.mse[row, col], one.mse[0, 0])
            assert_allclose(r.bias[row, col], one.bias[0, 0])
            assert_allclose(r.variance[row, col], one.variance[0, 0])
            assert r.converged[row, col] == one.converged[0, 0]
    assert_array_equal(r.bandwidths, bw)
    assert not r.bandwidths.flags.writeable


def test_selection_and_fitted_hazard_use_selected_smoothed_radii():
    t = np.linspace(0.1, 3, 20)
    r = muhaz_knn(t, neighbors=[2, 4, 3], bounds=(0, 3), n_min_grid=9, n_est_grid=13)
    assert_array_equal(r.scores, r.diagnostics.mse.sum(axis=1))
    assert r.selected_index == r.scores.argmin()
    assert r.neighbors == r.neighbor_bandwidths.neighbors[r.selected_index]
    for i in [0, 5, 12]:
        fixed = muhaz_fixed(t, bandwidth=r.bandwidth[i], grid=r.time[i : i + 1])
        assert_allclose(r.hazard[i], fixed.hazard[0])
    for array in (r.time, r.hazard, r.bandwidth, r.scores):
        assert not array.flags.writeable


def test_defaults_and_single_neighbor_bypass_with_subset():
    t = np.arange(1.0, 22.0)
    r = muhaz_knn(t, n_min_grid=7, n_est_grid=9)
    assert_array_equal(r.neighbor_bandwidths.neighbors, np.arange(2, 11))
    assert r.bounds == (0, 12)
    args = dict(
        times=[1, 2, 3, np.nan],
        delta=[1, 0, 1, np.nan],
        subset=[True, True, True, False],
        neighbors=2,
        bounds=(0, 4),
    )
    one = muhaz_knn(**args, smoothing_bandwidth=4)
    assert one.bounds == (0, 3)
    assert one.n_observations == 3 and one.n_events == 2
    assert one.diagnostics is one.scores is one.score is None
    assert one.bandwidth.shape == one.hazard.shape == (101,)


def test_time_scaling_and_neighbor_order():
    t = np.linspace(0.1, 3, 15)
    args = dict(
        neighbors=[2, 3],
        bounds=(0, 3),
        pilot_bandwidth=0.65,
        smoothing_bandwidth=1.3,
        n_min_grid=7,
        n_est_grid=9,
    )
    a = muhaz_knn(t, **args)
    args.update(neighbors=[3, 2], bounds=(0, 6), pilot_bandwidth=1.3, smoothing_bandwidth=2.6)
    b = muhaz_knn(t * 2, **args)
    assert a.neighbors == b.neighbors
    assert_allclose(a.scores / 4, b.scores[::-1])
    assert_allclose(a.bandwidth * 2, b.bandwidth)
    assert_allclose(a.hazard / 2, b.hazard)


def test_zero_radius_rejected_before_mse():
    with pytest.raises(ValueError, match="Zero neighbor radii"):
        muhaz_knn([1, 1, 2, 3], neighbors=[1, 2], method="failures", bounds=(0, 3), n_min_grid=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neighbors": None},
        {"neighbors": 0},
        {"smoothing_bandwidth": 0},
        {"smoothing_bandwidth": np.inf},
        {"kernel": "unknown"},
        {"boundary": "unknown"},
    ],
)
def test_invalid_fit_settings(kwargs):
    args = dict(times=[1, 2, 3], neighbors=2, bounds=(0, 3))
    args.update(kwargs)
    with pytest.raises(ValueError):
        muhaz_knn(**args)


def test_legacy_score_cutoff_has_explicit_failure_instead_of_undefined_count():
    t = np.array([0.2, 0.4, 0.8, 1.1, 1.7, 2.1, 2.4, 2.8, 3]) * 0.001
    args = dict(
        neighbors=[2, 3, 4],
        method="failures",
        pilot_bandwidth=0.00065,
        smoothing_bandwidth=0.0013,
        bounds=(0, 0.003),
        n_min_grid=9,
        n_est_grid=25,
    )
    r = muhaz_knn(t, **args)
    assert np.all(r.scores > 1e5)
    assert np.isfinite(r.hazard).all()
    with pytest.raises(RuntimeError, match="no defined optimum"):
        muhaz_knn(t, **args, legacy=True)
