"""Native fixed-bandwidth hazard curves and independent kernel identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import muhaz_fixed
from mdanderson_stats.muhaz import _kernel

KERNELS = ["rectangle", "epanechnikov", "biquadratic", "triquadratic"]
BOUNDARIES = ["none", "left", "both"]
CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-fixed.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_hazden(case):
    result = muhaz_fixed(
        case["times"],
        case["delta"],
        bandwidth=case["bandwidth"],
        grid=case["grid"],
        bounds=(0, 3),
        kernel=KERNELS[case["kernel"]],
        boundary=BOUNDARIES[case["boundary"]],
        legacy=True,
    )
    assert_allclose(result.hazard, case["hazard"], rtol=2e-12, atol=3e-14)


@pytest.mark.parametrize("shape", range(4))
@pytest.mark.parametrize("q", [0, 0.2, 0.7, 1])
def test_boundary_kernel_has_unit_mass_and_zero_first_moment(shape, q):
    nodes, weights = np.polynomial.legendre.leggauss(12)
    u = (nodes + 1) * (q + 1) / 2 - 1
    w = weights * (q + 1) / 2
    k = _kernel(u, np.array(q), shape)
    assert_allclose(w @ k, 1, atol=2e-14)
    assert_allclose(w @ (u * k), 0, atol=2e-14)


def test_grouped_ties_use_all_tied_censoring_at_risk():
    result = muhaz_fixed(
        [1, 1, 1, 2], [1, 1, 0, 0], bandwidth=0.25, grid=[1], kernel="rectangle", boundary="none"
    )
    assert_allclose(result.hazard, [(2 / 4) * (0.5 / 0.25)])
    perm = muhaz_fixed(
        [1, 2, 1, 1], [0, 0, 1, 1], bandwidth=0.25, grid=[1], kernel="rectangle", boundary="none"
    )
    assert_array_equal(result.hazard, perm.hazard)
    legacy = muhaz_fixed(
        [1, 1, 1, 2],
        [1, 1, 0, 0],
        bandwidth=0.25,
        grid=[1],
        kernel="rectangle",
        boundary="none",
        legacy=True,
    )
    assert_allclose(legacy.hazard, [(1 / 4 + 1 / 3) * 2])


def test_closed_rectangle_support_and_source_endpoint_difference():
    args = dict(bandwidth=1, grid=[1], kernel="rectangle", boundary="none")
    ordinary = muhaz_fixed([0, 2, 3], [1, 1, 0], **args)
    archived = muhaz_fixed([0, 2, 3], [1, 1, 0], legacy=True, **args)
    assert_allclose(ordinary.hazard, [(1 / 3 + 1 / 2) / 2])
    assert_array_equal(archived.hazard, [0])


def test_time_scaling_inverse_hazard_and_permutation():
    t = np.array([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3])
    d = np.array([1, 0, 1, 1, 0, 1, 0])
    first = muhaz_fixed(t, d, bandwidth=0.75)
    second = muhaz_fixed(t[::-1] * 5, d[::-1], bandwidth=0.75 * 5)
    assert_allclose(second.time, first.time * 5)
    assert_allclose(second.hazard, first.hazard / 5, atol=1e-15)
    with pytest.raises(ValueError):
        first.hazard[0] = 9


def test_empty_queries_all_censored_and_default_events():
    assert muhaz_fixed([1, 2], bandwidth=0.5, grid=[]).hazard.shape == (0,)
    assert_array_equal(muhaz_fixed([1, 2], [0, 0], bandwidth=0.5).hazard, 0)
    assert_array_equal(
        muhaz_fixed([1, 2], bandwidth=0.5).hazard, muhaz_fixed([1, 2], [1, 1], bandwidth=0.5).hazard
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"times": []},
        {"times": [-1, 2]},
        {"times": [np.nan, 2]},
        {"delta": [1]},
        {"delta": [1, 2]},
        {"bandwidth": 0},
        {"bandwidth": np.inf},
        {"grid": [-1]},
        {"grid": [[1]]},
        {"bounds": (1, 1)},
        {"kernel": "unknown"},
        {"boundary": "unknown"},
        {"legacy": 1},
    ],
)
def test_invalid_inputs(overrides):
    args = dict(times=[1, 2], bandwidth=0.5)
    args.update(overrides)
    with pytest.raises(ValueError):
        muhaz_fixed(**args)


def test_negative_boundary_kernel_total_is_truncated():
    result = muhaz_fixed([0.9], bandwidth=1, bounds=(0, 2), grid=[0], kernel="rectangle")
    assert_array_equal(result.hazard, [0])


def test_large_grid_agrees_with_individual_queries():
    times = np.linspace(0.01, 5, 1000)
    grid = np.linspace(0, 5, 700)
    result = muhaz_fixed(times, bandwidth=0.3, grid=grid)
    for index in np.random.default_rng(981).choice(grid.size, 12, replace=False):
        single = muhaz_fixed(times, bandwidth=0.3, grid=[grid[index]])
        assert_allclose(result.hazard[index], single.hazard[0], rtol=2e-14)
