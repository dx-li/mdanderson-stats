"""Source-pinned random survival forest kernels and deterministic trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mdanderson_stats.random_survival_forest import (
    _leaf_curve,
    _logrank_score,
    _time_grid,
    fit_random_survival_forest,
    predict_random_survival_forest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "random-survival-forest-native.json"


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
