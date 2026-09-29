"""Source-pinned random survival forest kernels and deterministic trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import (
    _factor_split_candidates,
    _factor_split_plan,
    _leaf_curve,
    _logrank_score,
    _PackedTree,
    _time_grid,
    _tree_goes_left,
    fit_random_survival_forest,
    predict_random_survival_forest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "random-survival-forest-native.json"


def test_shared_tree_router_preserves_numeric_less_equal_split() -> None:
    tree = _PackedTree(
        feature=np.asarray([0, -1, -1]),
        threshold=np.asarray([1.5, np.nan, np.nan]),
        categorical_node=np.asarray([False, False, False]),
        split_level_offset=np.zeros(3, dtype=np.int32),
        split_level_count=np.zeros(3, dtype=np.int32),
        split_levels=np.empty(0, dtype=np.int32),
        left=np.asarray([1, -1, -1]),
        right=np.asarray([2, -1, -1]),
        event_offset=np.zeros(3, dtype=np.int64),
        event_count=np.zeros(3, dtype=np.int64),
        event_time=np.empty(0),
        log_survival=np.empty(0),
        cumulative_hazard=np.empty(0),
    )
    assert _tree_goes_left(tree, 0, 1.5)
    assert _tree_goes_left(tree, 0, np.nextafter(1.5, -np.inf))
    assert not _tree_goes_left(tree, 0, np.nextafter(1.5, np.inf))


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_native_leaf_kaplan_meier_nelson_aalen_and_logrank_kernels() -> None:
    fixture = _fixture()
    for case in fixture["cases"]:
        time = np.asarray(case["time"], dtype=np.float64)
        event = np.asarray(case["event"], dtype=np.float64)
        query = np.asarray(case["times"], dtype=np.float64)
        event_times, log_survival, cumulative_hazard = _leaf_curve(time, event)
        step_index = np.searchsorted(event_times, query, side="right") - 1
        survival = np.ones(query.size)
        hazard = np.zeros(query.size)
        included = step_index >= 0
        survival[included] = np.exp(log_survival[step_index[included]])
        hazard[included] = cumulative_hazard[step_index[included]]
        np.testing.assert_allclose(survival, case["survival"], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(hazard, case["cumulative_hazard"], rtol=2e-14, atol=2e-14)

        x = np.asarray(case["x"], dtype=np.float64)
        for split in case["splits"]:
            left = x <= float(split["cut"])
            score = _logrank_score(time, event, left)
            np.testing.assert_allclose(score, float(split["score"]), rtol=2e-14, atol=2e-14)


def test_native_randomforestsrc_time_grids() -> None:
    fixture = _fixture()
    for reference in fixture["time_grids"]:
        event_times = np.asarray(reference["event_times"], dtype=np.float64)
        actual = _time_grid(event_times, int(reference["ntime"]))
        np.testing.assert_array_equal(actual, reference["expected"])
    np.testing.assert_array_equal(_time_grid(np.array([1.0, 2.0, 3.0]), [0.5, 2.5, 9]), [1, 2, 3])
    np.testing.assert_array_equal(_time_grid(np.array([1.0, 2.0, 3.0]), np.array([2])), [1, 3])


def test_native_exhaustive_full_sample_trees_and_survival_surfaces() -> None:
    fixture = _fixture()
    cases = {case["name"]: case for case in fixture["cases"]}
    for reference in fixture["trees"]:
        case = cases[reference["case"]]
        fit = fit_random_survival_forest(
            case["time"],
            case["event"],
            case["x"],
            n_trees=1,
            mtry=1,
            nodesize=int(reference["nodesize"]),
            nsplit=0,
            sample_fraction=1.0,
            ntime=0,
            random_state=17,
        )
        tree = fit.trees[0]
        native_nodes = reference["nodes"]
        assert tree.feature.size == len(native_nodes)
        assert tree.feature.dtype.kind in "iu"
        assert tree.left.dtype.kind in "iu"
        node_pairs = [(0, 0)]
        while node_pairs:
            index, native_index = node_pairs.pop()
            native_node = native_nodes[native_index]
            if native_node["cut"] is None:
                assert tree.feature[index] == -1
            else:
                assert tree.feature[index] == 0
                assert tree.threshold[index] == native_node["cut"]
                node_pairs.append((int(tree.left[index]), int(native_node["left"])))
                node_pairs.append((int(tree.right[index]), int(native_node["right"])))
        prediction = predict_random_survival_forest(
            fit,
            case["times"],
            np.asarray(reference["profiles"], dtype=np.float64)[:, None],
        )
        np.testing.assert_allclose(
            prediction.survival, reference["survival"], rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            prediction.cumulative_hazard,
            reference["cumulative_hazard"],
            rtol=2e-14,
            atol=2e-14,
        )


def test_seeded_forest_repeats_and_bounds_predictions() -> None:
    cases = _fixture()["cases"]
    case = cases[0]
    arguments = (
        case["time"],
        case["event"],
        case["x"],
    )
    first = fit_random_survival_forest(*arguments, n_trees=8, random_state=231)
    second = fit_random_survival_forest(*arguments, n_trees=8, random_state=231)
    profiles = np.array([[-1.0], [2.0]])
    prediction = predict_random_survival_forest(first, [0, 1, 3, 8, 20], profiles)
    repeated = predict_random_survival_forest(second, [0, 1, 3, 8, 20], profiles)
    np.testing.assert_array_equal(prediction.survival, repeated.survival)
    np.testing.assert_array_equal(prediction.cumulative_hazard, repeated.cumulative_hazard)
    assert np.all(np.diff(prediction.survival, axis=1) <= 0)
    assert np.all(np.diff(prediction.cumulative_hazard, axis=1) >= 0)
    assert np.all(prediction.survival <= 1)
    replacement = fit_random_survival_forest(
        *arguments,
        n_trees=1,
        replace=True,
        sample_fraction=0.5,
        random_state=231,
    )
    assert replacement.sampled_rows == int(np.rint(0.5 * len(case["time"])))

    zero_time_case = next(item for item in cases if item["name"] == "zero_time")
    root_fit = fit_random_survival_forest(
        zero_time_case["time"],
        zero_time_case["event"],
        zero_time_case["x"],
        n_trees=1,
        nodesize=20,
        sample_fraction=1.0,
        ntime=0,
        random_state=7,
    )
    root_prediction = predict_random_survival_forest(
        root_fit,
        zero_time_case["times"],
        np.asarray(zero_time_case["x"], dtype=np.float64)[:, None],
    )
    np.testing.assert_allclose(
        root_prediction.survival,
        np.broadcast_to(zero_time_case["survival"], root_prediction.survival.shape),
        rtol=2e-14,
        atol=2e-14,
    )


def test_unordered_categorical_subset_routing_and_unknown_levels() -> None:
    categories = np.repeat([10.0, 20.0, 30.0, 40.0], 4)
    early = np.isin(categories, [10.0, 30.0])
    time = np.tile(np.arange(1.0, 5.0), 4) + np.where(early, 0.0, 10.0)
    fit = fit_random_survival_forest(
        time,
        np.ones(time.size),
        categories,
        categorical_features=[0],
        n_trees=1,
        mtry=1,
        nodesize=1,
        nsplit=10,
        sample_fraction=1.0,
        ntime=0,
        random_state=71,
    )
    tree = fit.trees[0]
    assert tree.categorical_node[0]
    assert fit.categorical_levels[0] is not None
    assert fit.covariate_mean[0] == 10.0  # tied mode resolves to smallest label
    codes = np.searchsorted(fit.categorical_levels[0], [10, 20, 30, 40])
    routed = [_tree_goes_left(tree, 0, float(code)) for code in codes]
    assert routed[0] == routed[2] and routed[1] == routed[3]
    assert routed[0] != routed[1]
    prediction = predict_random_survival_forest(fit, [0, 5, 15], [10.0])
    assert prediction.profile[0, 0] == 10.0
    assert np.all(np.isfinite(prediction.survival))
    with pytest.raises(ValueError, match="unseen level"):
        predict_random_survival_forest(fit, [1, 2], [99.0])
    oob_fit = fit_random_survival_forest(
        time,
        np.ones(time.size),
        categories,
        categorical_features=[0],
        n_trees=4,
        nodesize=1,
        sample_fraction=0.5,
        compute_oob=True,
        ntime=0,
        random_state=22,
    )
    assert oob_fit.oob is not None
    assert np.any(oob_fit.oob.contributor_count > 0)


def test_factor_partition_planning_is_bounded_and_respects_native_exact_boundary() -> None:
    levels = np.arange(100, dtype=np.float64)
    assert _factor_split_plan(levels[:4], node_size=7, nsplit=7) == (7, True, ())
    candidate_count, exact, probabilities = _factor_split_plan(
        levels, node_size=100, nsplit=10
    )
    assert candidate_count == 10 and not exact
    candidates = list(
        _factor_split_candidates(
            levels, candidate_count, exact, probabilities, np.random.default_rng(17)
        )
    )
    assert len(candidates) == 10
    assert all(1 <= subset.size <= 50 for subset, _ in candidates)


def test_contour_keeps_nominal_adjustment_at_its_mode() -> None:
    from mdanderson_stats.random_survival_forest_contour import random_survival_forest_contour

    category = np.tile(np.repeat([2.0, 7.0], 8), 2)
    continuous = np.repeat(np.arange(1.0, 5.0), 8)
    time = np.arange(1.0, category.size + 1.0)
    x = np.column_stack((category, continuous))
    fit = fit_random_survival_forest(
        time,
        np.ones(time.size),
        x,
        categorical_features=[0],
        n_trees=2,
        nodesize=2,
        ntime=0,
        random_state=9,
    )
    reference = x[category == 7.0]
    contour = random_survival_forest_contour(
        fit, reference, 1, grid=[1.0, 2.0, 4.0], times=[0.0, 10.0]
    )
    assert np.all(contour.profile[0] == 7.0)
    assert np.all(contour.quantile_profiles[:, 0] == 7.0)
    with pytest.raises(ValueError, match="cannot be a categorical"):
        random_survival_forest_contour(fit, x, 0, grid=[2, 7])
