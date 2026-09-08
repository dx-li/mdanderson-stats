"""Native global selection and independent settings/minimization contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import muhaz_fixed, muhaz_global

CASES = json.loads((Path(__file__).parent / "fixtures/muhaz-global.json").read_text())["cases"]
KERNELS = ["rectangle", "epanechnikov", "biquadratic", "triquadratic"]
BOUNDARIES = ["none", "left", "both"]


@pytest.mark.parametrize("case", CASES)
def test_original_newhad_global(case):
    r = muhaz_global(
        case["times"],
        case["delta"],
        bandwidths=case["bandwidths"],
        pilot_bandwidth=0.65,
        bounds=(0, 3),
        n_min_grid=9,
        n_est_grid=25,
        kernel=KERNELS[case["kernel"]],
        boundary=BOUNDARIES[case["boundary"]],
        legacy=True,
    )
    assert r.bandwidth == case["bandwidth"]
    assert_allclose(r.hazard, case["hazard"], rtol=2e-11, atol=2e-13)
    if case["scores"] is None:
        assert r.scores is r.score is r.diagnostics is None
    else:
        assert_allclose(r.scores, case["scores"], rtol=2e-10, atol=3e-12)
        assert_allclose(r.score, case["score"], rtol=2e-10)


def test_selection_minimizes_diagnostic_grid_sum_and_matches_selected_fixed_fit():
    r = muhaz_global(
        [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3],
        [1, 0, 1, 1, 0, 1, 0],
        bandwidths=[0.35, 0.8, 2],
        pilot_bandwidth=0.65,
        bounds=(0, 3),
    )
    assert_allclose(r.scores, r.diagnostics.mse.sum(axis=1))
    assert r.selected_index == np.argmin(r.scores)
    assert r.score == r.scores.min()
    fixed = muhaz_fixed(
        [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1, 0], bandwidth=r.bandwidth
    )
    assert_array_equal(r.hazard, fixed.hazard)


def test_ten_at_risk_bound_uses_linear_interpolation_at_distinct_times():
    times = np.repeat(np.arange(1, 8), 3)
    r = muhaz_global(times, bandwidths=0.5)
    # Risks 12 and 9 at times 4 and 5 bracket ten at risk.
    assert_allclose(r.curve.bounds, [0, 4 + 2 / 3])
    assert r.n_observations == r.n_events == 21
    assert r.diagnostics is None


def test_default_pilot_and_grid_include_span_correction_for_nonzero_lower_bound():
    times = np.linspace(0.1, 4, 20)
    r = muhaz_global(times, bounds=(1, 3), n_min_grid=5, n_est_grid=7)
    expected = 2 / (8 * 20**0.2)
    assert r.diagnostics.pilot_bandwidth == expected
    assert_allclose(r.bandwidths, np.linspace(0.2 * expected, 20 * expected, 25))
    assert r.hazard.size == 7
    legacy = muhaz_global(times, bounds=(1, 3), n_min_grid=5, n_est_grid=7, legacy=True)
    assert legacy.diagnostics.pilot_bandwidth == 3 / (8 * 20**0.2)


def test_subset_precedes_selected_data_validation_and_bounds_are_clamped():
    r = muhaz_global(
        [1, 2, np.nan], [1, 0, np.nan], subset=[True, True, False], bandwidths=0.4, bounds=(0, 5)
    )
    expected = muhaz_fixed([1, 2], [1, 0], bandwidth=0.4)
    assert_array_equal(r.hazard, expected.hazard)
    assert r.curve.bounds == (0, 2)
    assert r.n_observations == 2 and r.n_events == 1


def test_zero_scores_and_equal_candidates_have_explicit_selection():
    args = dict(
        times=[1, 2], delta=[0, 0], bandwidths=[0.2, 0.4], pilot_bandwidth=0.3, bounds=(0, 2)
    )
    r = muhaz_global(**args)
    assert r.selected_index == 0 and r.score == 0
    legacy = muhaz_global(**args, legacy=True)
    assert legacy.selected_index == 1 and legacy.score == 1e30
    equal = muhaz_global([1, 2, 3], bandwidths=[0.5, 0.5], pilot_bandwidth=0.4, bounds=(0, 3))
    assert equal.selected_index == 0
    with pytest.raises(ValueError):
        r.bandwidths[0] = 9


def test_grid_exceeding_original_static_pilot_buffer_is_supported():
    r = muhaz_global(
        np.linspace(0.1, 2, 20),
        bandwidths=[0.2, 0.4],
        pilot_bandwidth=0.3,
        bounds=(0, 2),
        n_min_grid=1001,
        n_est_grid=3,
    )
    assert r.diagnostics.mse.shape == (2, 1001)
    assert np.all(np.isfinite(r.hazard))


@pytest.mark.parametrize(
    "overrides",
    [
        {"times": []},
        {"times": [-1, 2]},
        {"delta": [1]},
        {"delta": [1, 2]},
        {"bounds": None},
        {"bounds": (2, 1)},
        {"subset": [1, 0]},
        {"subset": [False, False]},
        {"bandwidths": []},
        {"bandwidths": [-1]},
        {"pilot_bandwidth": 0},
        {"n_min_grid": 0},
        {"n_est_grid": 1.5},
        {"legacy": 1},
        {"bandwidths": None, "delta": [0, 0], "pilot_bandwidth": None},
    ],
)
def test_invalid_inputs(overrides):
    args = dict(times=[1, 2], bandwidths=[0.2, 0.4], pilot_bandwidth=0.3, bounds=(0, 2))
    args.update(overrides)
    with pytest.raises(ValueError):
        muhaz_global(**args)
