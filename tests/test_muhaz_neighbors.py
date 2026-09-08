"""Native radii, order statistics and survival-mass invariants."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.muhaz_neighbors import muhaz_neighbor_bandwidths

CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-neighbors.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_neighbor_bandwidths(case):
    r = muhaz_neighbor_bandwidths(
        case["times"],
        case["delta"],
        neighbors=case["neighbors"],
        grid=case["grid"],
        method="failures" if case["method"] == 1 else "survival",
        legacy=True,
    )
    assert_allclose(r.bandwidth[0], case["bandwidth"], rtol=2e-13, atol=2e-14)


def test_failure_distances_ignore_censoring_and_preserve_requested_order():
    r = muhaz_neighbor_bandwidths(
        [1, 2, 3, 4, 5], [1, 0, 1, 0, 1], neighbors=[3, 1, 2, 1], grid=[0, 3, 7], method="failures"
    )
    assert_array_equal(r.bandwidth, [[5, 2, 6], [1, 0, 2], [3, 2, 4], [1, 0, 2]])
    assert_array_equal(r.neighbors, [3, 1, 2, 1])
    for a in [r.time, r.neighbors, r.bandwidth]:
        assert not a.flags.writeable


def test_uncensored_survival_mass_uses_radius_endpoint_conventions():
    r = muhaz_neighbor_bandwidths([1, 2, 3, 4], neighbors=[1, 2, 3], grid=[0, 5])
    # Right endpoint enters the survival difference; left endpoint is excluded.
    assert_allclose(r.bandwidth[:, 0], np.array([1, 2, 3]) * 0.99999)
    assert_allclose(r.bandwidth[:, 1], [1, 2, 3])


def test_terminal_singleton_failure_is_included_by_default():
    args = dict(times=[1, 2, 3, 4], neighbors=2, grid=[4])
    corrected = muhaz_neighbor_bandwidths(**args)
    original = muhaz_neighbor_bandwidths(**args, legacy=True)
    assert_allclose(corrected.bandwidth, [[1]])
    assert_allclose(original.bandwidth, [[2]])


def test_terminal_tie_is_retained_in_both_modes():
    args = dict(times=[1, 2, 3, 4, 4], delta=[1, 0, 1, 0, 1], neighbors=[2, 3], grid=[0, 2, 5])
    a = muhaz_neighbor_bandwidths(**args)
    b = muhaz_neighbor_bandwidths(**args, legacy=True)
    assert_array_equal(a.bandwidth, b.bandwidth)


def test_one_subject_and_empty_legacy_table():
    r = muhaz_neighbor_bandwidths([2], neighbors=1, grid=[0])
    assert_allclose(r.bandwidth, [[1.99998]])
    with pytest.raises(ValueError, match="table is empty"):
        muhaz_neighbor_bandwidths([2], neighbors=1, grid=[0], legacy=True)


@pytest.mark.parametrize("method", ["failures", "survival"])
def test_scaling_permutation_and_empty_grid(method):
    t = np.array([0.2, 0.5, 1.1, 1.1, 2.3, 3, 4])
    d = np.array([1, 0, 1, 0, 1, 1, 0])
    grid = np.array([0, 0.7, 2.1, 5])
    a = muhaz_neighbor_bandwidths(t, d, neighbors=[2, 3], grid=grid, method=method)
    b = muhaz_neighbor_bandwidths(
        t[::-1] * 2, d[::-1], neighbors=[3, 2], grid=grid * 2, method=method
    )
    assert_allclose(a.bandwidth * 2, b.bandwidth[::-1])
    empty = muhaz_neighbor_bandwidths(t, d, neighbors=[2, 3], grid=[], method=method)
    assert empty.bandwidth.shape == (2, 0)


def test_failure_sample_larger_than_archived_buffers_and_chunk_boundaries():
    t = np.arange(1.0, 25002.0)
    grid = np.linspace(0, 25002, 31)
    r = muhaz_neighbor_bandwidths(t, neighbors=[1, 5, 100], grid=grid, method="failures")
    for i in [0, 9, 10, 19, 20, 30]:
        expected = np.sort(np.abs(t - grid[i]))[[0, 4, 99]]
        assert_allclose(r.bandwidth[:, i], expected)


def test_survival_batch_crosses_chunk_boundaries():
    t = np.arange(1.0, 501.0)
    grid = np.linspace(0, 600, 4000)
    r = muhaz_neighbor_bandwidths(t, neighbors=50, grid=grid)
    for i in [0, 2594, 2595, 3999]:
        one = muhaz_neighbor_bandwidths(t, neighbors=50, grid=grid[i : i + 1])
        assert_array_equal(r.bandwidth[:, i], one.bandwidth[:, 0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"times": []},
        {"times": [-1, 2]},
        {"times": [1, np.nan]},
        {"delta": [1]},
        {"delta": [0, 0]},
        {"delta": [1, 2]},
        {"neighbors": []},
        {"neighbors": 0},
        {"neighbors": 3},
        {"neighbors": 1.5},
        {"neighbors": [[1]]},
        {"grid": [-1]},
        {"grid": [[1]]},
        {"grid": [np.inf]},
        {"method": "unknown"},
        {"legacy": 1},
    ],
)
def test_invalid_inputs(overrides):
    args = dict(times=[1, 2], neighbors=1, grid=[0, 1])
    args.update(overrides)
    with pytest.raises(ValueError):
        muhaz_neighbor_bandwidths(**args)
