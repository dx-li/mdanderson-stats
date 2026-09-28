"""Explicit OOB permutation importance conventions and bounded fit matching."""

from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import fit_random_survival_forest
from mdanderson_stats.random_survival_forest_vimp import (
    permutation_random_survival_forest_importance,
)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 30
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 3 != 0).astype(float)
    x = np.column_stack((np.sin(np.arange(n)), np.ones(n)))
    return time, event, x


def test_constant_feature_has_zero_permutation_importance() -> None:
    time, event, x = _data()
    fit = fit_random_survival_forest(
        time,
        event,
        x,
        n_trees=12,
        mtry=2,
        nodesize=2,
        nsplit=0,
        sample_fraction=0.4,
        ntime=0,
        random_state=923,
        compute_oob=True,
    )
    result = permutation_random_survival_forest_importance(
        fit, time, event, x, feature_indices=[1], random_state=5
    )
    assert result.block_count == 1
    assert result.valid_block_count.tolist() == [1]
    assert result.importance[0] == 0.0
    assert result.block_importance[0, 0] == 0.0
    np.testing.assert_array_equal(result.ignored_tree_indices, [])


def test_complete_block_tail_is_reported_and_rng_replays() -> None:
    time, event, x = _data()
    fit = fit_random_survival_forest(
        time,
        event,
        x,
        n_trees=8,
        mtry=2,
        nodesize=2,
        nsplit=0,
        sample_fraction=0.4,
        ntime=0,
        random_state=73,
        compute_oob=True,
    )
    first = permutation_random_survival_forest_importance(
        fit, time, event, x, feature_indices=[0], block_size=3, random_state=91
    )
    again = permutation_random_survival_forest_importance(
        fit, time, event, x, feature_indices=[0], block_size=3, random_state=91
    )
    assert first.block_count == 2
    np.testing.assert_array_equal(first.ignored_tree_indices, [6, 7])
    np.testing.assert_array_equal(first.block_importance, again.block_importance)
    np.testing.assert_array_equal(first.baseline_error, again.baseline_error)
    assert not first.feature_indices.flags.writeable


def test_no_oob_pair_and_training_identity_are_explicit() -> None:
    time, event, x = _data()
    fit = fit_random_survival_forest(
        time,
        event,
        x,
        n_trees=3,
        mtry=1,
        nodesize=2,
        nsplit=0,
        sample_fraction=1.0,
        random_state=4,
        compute_oob=True,
    )
    result = permutation_random_survival_forest_importance(
        fit, time, event, x, feature_indices=[0], random_state=2
    )
    assert result.valid_block_count.tolist() == [0]
    assert np.isnan(result.importance[0])
    with pytest.raises(ValueError, match="row order"):
        permutation_random_survival_forest_importance(
            fit, time[::-1], event[::-1], x[::-1], feature_indices=[0], random_state=2
        )
    with pytest.raises(ValueError, match="exactly one"):
        permutation_random_survival_forest_importance(fit, time, event, x, feature_indices=[0])
