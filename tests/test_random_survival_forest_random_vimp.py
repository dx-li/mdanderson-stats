"""Source-defined random split routing and OOB block importance."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats.random_survival_forest import _PackedTree, fit_random_survival_forest
from mdanderson_stats.random_survival_forest_vimp import (
    _route_random_split_hazard,
    random_split_random_survival_forest_importance,
)


class _FixedDraw:
    def __init__(self, value: float) -> None:
        self.value = value
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return self.value


def _two_leaf_tree(node_count: int, left_count: int) -> _PackedTree:
    return _PackedTree(
        feature=np.array([0, -1, -1], dtype=np.int32),
        threshold=np.array([0.5, 0.0, 0.0]),
        categorical_node=np.array([False, False, False]),
        split_level_offset=np.array([0, 0, 0], dtype=np.int64),
        split_level_count=np.array([0, 0, 0], dtype=np.int64),
        split_levels=np.array([], dtype=np.int64),
        left=np.array([1, -1, -1], dtype=np.int32),
        right=np.array([2, -1, -1], dtype=np.int32),
        event_offset=np.array([0, 0, 1], dtype=np.int64),
        event_count=np.array([0, 1, 1], dtype=np.int64),
        event_time=np.array([1.0, 1.0]),
        log_survival=np.array([0.0, -2.0, -4.0]),
        cumulative_hazard=np.array([2.0, 4.0]),
        represented_count=np.array([node_count, left_count, node_count - left_count]),
    )


def test_random_split_routes_match_unchanged_native_kernel_cases() -> None:
    cases_path = Path(__file__).parent / "fixtures" / "random-survival-forest-random-routing.csv"
    native_path = (
        Path(__file__).parent / "fixtures" / "random-survival-forest-random-routing-native.csv"
    )
    with native_path.open(newline="") as stream:
        native = list(csv.DictReader(stream))
    assert len(native) == 15
    assert all(row["match"] == "true" for row in native)
    native_by_id = {row["case_id"]: row for row in native}
    with cases_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["scope"] == "importance_flag_group":
            continue  # Group flags are outside this per-feature API.
        split_feature = int(row["split_feature"]) - 1
        target_feature = int(row["target_feature"]) - 1
        tree = _two_leaf_tree(int(row["node_count"]), int(row["left_count"]))
        if row["scope"] == "terminal_node":
            tree = replace(
                tree,
                feature=np.array([-1, -1, -1], dtype=np.int32),
                left=np.array([-1, -1, -1], dtype=np.int32),
                right=np.array([-1, -1, -1], dtype=np.int32),
                event_count=np.array([0, 1, 1], dtype=np.int64),
            )
        profile = np.array([0.25 if int(row["ordinary_branch"]) == 1 else 0.75, 0.25])
        draw = _FixedDraw(float(row["alpha"]))
        hazard = _route_random_split_hazard(
            tree,
            profile,
            np.array([1.0]),
            target_feature,
            threshold=float(row["threshold"]),
            rng=draw,  # type: ignore[arg-type]
        )
        terminal = int(row["expected_terminal"])
        # The small tree maps its left/right leaves to distinct hazards.
        expected_hazard = {101: 2.0, 102: 4.0, 103: 0.0}[terminal]
        assert hazard == expected_hazard, row["case_id"]
        assert draw.count == int(row["expected_draws"]), row["case_id"]
        native_row = native_by_id[row["case_id"]]
        assert native_row["actual_terminal"] == row["expected_terminal"]
        assert int(native_row["actual_draws"]) == draw.count
        assert split_feature == 0


def test_oob_fit_retains_bootstrap_counts_and_importance_replays() -> None:
    n = 30
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 3 != 0).astype(float)
    covariates = np.column_stack((np.sin(np.arange(n)), np.ones(n)))
    fit = fit_random_survival_forest(
        time,
        event,
        covariates,
        n_trees=8,
        mtry=2,
        nodesize=2,
        nsplit=0,
        sample_fraction=0.8,
        replace=True,
        ntime=0,
        random_state=73,
        compute_oob=True,
    )
    assert all(tree.represented_count is not None for tree in fit.trees)
    assert fit.inbag_membership is not None
    saw_replacement_duplicate = False
    for tree_index, tree in enumerate(fit.trees):
        counts = tree.represented_count
        assert counts is not None
        saw_replacement_duplicate |= int(
            np.unpackbits(fit.inbag_membership[tree_index], bitorder="little")[:n].sum()
        ) < int(counts[0])
        assert not counts.flags.writeable
        internal = tree.feature >= 0
        np.testing.assert_array_equal(
            counts[internal], counts[tree.left[internal]] + counts[tree.right[internal]]
        )
        assert int(counts[0]) == fit.sampled_rows // fit.n_trees
    assert saw_replacement_duplicate
    first = random_split_random_survival_forest_importance(
        fit, time, event, covariates, feature_indices=[0], block_size=3, random_state=91
    )
    again = random_split_random_survival_forest_importance(
        fit, time, event, covariates, feature_indices=[0], block_size=3, random_state=91
    )
    np.testing.assert_array_equal(first.block_importance, again.block_importance)
    np.testing.assert_array_equal(first.perturbed_error, again.perturbed_error)
    assert first.block_count == 2
    np.testing.assert_array_equal(first.ignored_tree_indices, [6, 7])


def test_zero_threshold_keeps_oob_baseline_importance() -> None:
    n = 30
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 3 != 0).astype(float)
    covariates = np.column_stack((np.sin(np.arange(n)), np.ones(n)))
    fit = fit_random_survival_forest(
        time,
        event,
        covariates,
        n_trees=6,
        mtry=2,
        nodesize=2,
        nsplit=0,
        sample_fraction=0.5,
        ntime=0,
        random_state=73,
        compute_oob=True,
    )
    result = random_split_random_survival_forest_importance(
        fit,
        time,
        event,
        covariates,
        feature_indices=[0],
        vimp_threshold=0.0,
        random_state=91,
    )
    np.testing.assert_array_equal(result.block_importance, 0.0)


def test_zero_uniform_at_zero_threshold_keeps_ordinary_route() -> None:
    tree = _two_leaf_tree(10, 3)
    draw = _FixedDraw(0.0)
    hazard = _route_random_split_hazard(
        tree,
        np.array([0.25]),
        np.array([1.0]),
        0,
        threshold=0.0,
        rng=draw,  # type: ignore[arg-type]
    )
    assert hazard == 2.0
    assert draw.count == 1
