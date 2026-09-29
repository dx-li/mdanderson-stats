"""Source-defined anti-split routing and OOB block importance."""

from __future__ import annotations

import numpy as np

from mdanderson_stats.random_survival_forest import _PackedTree, fit_random_survival_forest
from mdanderson_stats.random_survival_forest_vimp import (
    _route_anti_hazard,
    _route_hazard,
    anti_split_random_survival_forest_importance,
    permutation_random_survival_forest_importance,
    random_split_random_survival_forest_importance,
)


def _two_leaf_tree() -> _PackedTree:
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
    )


class _ZeroDraw:
    def random(self) -> float:
        return 0.0


def test_anti_split_flips_only_target_feature_nodes() -> None:
    tree = _two_leaf_tree()
    profile = np.array([0.25])
    ordinary = _route_hazard(tree, profile, np.array([1.0]))
    unchanged = _route_anti_hazard(
        tree,
        profile,
        np.array([1.0]),
        0,
        threshold=0.0,
        rng=_ZeroDraw(),  # type: ignore[arg-type]
    )
    opposite = _route_anti_hazard(
        tree, profile, np.array([1.0]), 0, threshold=1.0, rng=np.random.default_rng(1)
    )
    assert ordinary == unchanged == 2.0
    assert opposite == 4.0


def test_anti_vimp_zero_threshold_and_seeded_block_replay() -> None:
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
        sample_fraction=0.4,
        ntime=0,
        random_state=73,
        compute_oob=True,
    )
    zero = anti_split_random_survival_forest_importance(
        fit,
        time,
        event,
        covariates,
        feature_indices=[0],
        block_size=3,
        vimp_threshold=0.0,
        random_state=91,
    )
    first = anti_split_random_survival_forest_importance(
        fit, time, event, covariates, feature_indices=[0], block_size=3, random_state=91
    )
    again = anti_split_random_survival_forest_importance(
        fit, time, event, covariates, feature_indices=[0], block_size=3, random_state=91
    )
    np.testing.assert_array_equal(zero.block_importance, 0.0)
    np.testing.assert_array_equal(first.block_importance, again.block_importance)
    np.testing.assert_array_equal(first.perturbed_error, again.perturbed_error)
    assert first.vimp_threshold == 1.0
    assert first.block_count == 2
    np.testing.assert_array_equal(first.ignored_tree_indices, [6, 7])


def test_categorical_label_recode_preserves_both_oob_importance_routes() -> None:
    n = 36
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 4 != 0).astype(float)
    raw = np.resize(np.array([10.0, 20.0, 40.0]), n)[:, None]
    recoded = raw * 7.0 + 3.0
    kwargs = dict(
        categorical_features=[0],
        n_trees=10,
        mtry=1,
        nodesize=2,
        nsplit=0,
        sample_fraction=0.5,
        ntime=0,
        random_state=821,
        compute_oob=True,
    )
    original_fit = fit_random_survival_forest(time, event, raw, **kwargs)
    recoded_fit = fit_random_survival_forest(time, event, recoded, **kwargs)
    for vimp in (
        anti_split_random_survival_forest_importance,
        permutation_random_survival_forest_importance,
        random_split_random_survival_forest_importance,
    ):
        first = vimp(original_fit, time, event, raw, feature_indices=[0], random_state=912)
        second = vimp(recoded_fit, time, event, recoded, feature_indices=[0], random_state=912)
        np.testing.assert_array_equal(first.baseline_error, second.baseline_error)
        np.testing.assert_array_equal(first.perturbed_error, second.perturbed_error)
        with np.testing.assert_raises_regex(ValueError, "training data values"):
            vimp(original_fit, time, event, recoded, feature_indices=[0], random_state=912)
