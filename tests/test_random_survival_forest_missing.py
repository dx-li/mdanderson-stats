from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import (
    _impute_node_values,
    _terminal_impute_value,
    fit_random_survival_forest,
    predict_random_survival_forest,
)


class _TapeRng:
    def __init__(self, indices: list[int], uniform: float = 0.0) -> None:
        self.indices = iter(indices)
        self.uniform = uniform

    def integers(self, high: int, size: int | None = None) -> int | np.ndarray:
        count = 1 if size is None else size
        draws = np.fromiter((next(self.indices) for _ in range(count)), dtype=np.int64)
        if np.any(draws >= high):
            raise AssertionError("tape index outside donor pool")
        return int(draws[0]) if size is None else draws

    def random(self) -> float:
        return self.uniform


def test_node_imputation_preserves_bootstrap_duplicate_donors_and_ancestor_fills() -> None:
    values = np.array([2.0, -7.0, 8.0, -9.0])
    missing = np.array([False, True, False, True])
    # Row zero appears twice in the bootstrap donor pool, so it has two tape slots.
    filled, donor_rows = _impute_node_values(
        values,
        missing,
        np.array([0, 0, 2]),
        np.array([1, 3]),
        rng=_TapeRng([1, 2]),
    )
    np.testing.assert_array_equal(donor_rows, [0, 2])
    np.testing.assert_array_equal(filled, [2.0, 8.0])

    retained, empty_pool = _impute_node_values(
        values,
        missing,
        np.array([], dtype=np.int64),
        np.array([1, 3]),
        rng=_TapeRng([]),
    )
    np.testing.assert_array_equal(empty_pool, [-1, -1])
    np.testing.assert_array_equal(retained, [-7.0, -9.0])


def test_terminal_imputation_uses_snapped_mean_mode_and_inbag_fallback() -> None:
    values = np.array([1.0, 3.0, 3.0, 4.0])
    missing = np.array([False, False, False, True])
    times = np.array([1.0, 3.0, 4.0])
    # Mean 2 is tied; source tie rule uses the lower endpoint when U <= .5.
    assert (
        _terminal_impute_value(
            values,
            missing,
            np.array([0, 1]),
            np.array([3]),
            kind="time",
            master_times=times,
            rng=_TapeRng([], uniform=0.25),
        )
        == 1.0
    )
    assert (
        _terminal_impute_value(
            values,
            missing,
            np.array([0, 1, 2]),
            np.array([3]),
            kind="categorical",
            master_times=times,
            rng=_TapeRng([]),
        )
        == 3.0
    )
    assert (
        _terminal_impute_value(
            values,
            missing,
            np.array([], dtype=np.int64),
            np.array([3]),
            kind="time",
            master_times=times,
            rng=_TapeRng([]),
        )
        == 4.0
    )


def test_omit_fit_maps_retained_rows_and_oob_back_to_original_input() -> None:
    time = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    x = np.arange(6.0)[:, None]
    fit = fit_random_survival_forest(
        time,
        event,
        x,
        na_action="omit",
        n_trees=3,
        random_state=19,
        compute_oob=True,
    )
    np.testing.assert_array_equal(fit.training_row_indices, [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(fit.oob.row_indices, [0, 1, 3, 4, 5])
    assert fit.na_action == "omit"
    assert fit.original_training_fingerprint is not None
    direct = fit_random_survival_forest(
        time[[0, 1, 3, 4, 5]],
        event[[0, 1, 3, 4, 5]],
        x[[0, 1, 3, 4, 5]],
        n_trees=3,
        random_state=19,
        compute_oob=True,
    )
    for retained, complete in zip(fit.trees, direct.trees, strict=True):
        np.testing.assert_array_equal(retained.feature, complete.feature)
        np.testing.assert_array_equal(retained.threshold, complete.threshold)


def test_node_local_imputation_routes_oob_and_prediction_without_mutating_profiles() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    time[4] = np.nan
    fit = fit_random_survival_forest(
        time,
        event,
        x,
        na_action="impute",
        n_trees=3,
        mtry=1,
        nodesize=1,
        nsplit=0,
        replace=False,
        sample_fraction=0.8,
        random_state=11,
        compute_oob=True,
    )
    assert fit.imputation_performed
    assert fit.requested_trees == fit.n_trees == 3
    assert fit.oob is not None and not fit.oob.concordance_available
    assert fit.oob.comparable_pairs == 0
    assert all(tree.training_leaf_node is not None for tree in fit.trees)
    contributing = fit.oob.contributor_count > 0
    assert np.any(contributing)
    assert np.all(np.isfinite(fit.oob.survival[contributing]))
    assert np.all(np.diff(fit.oob.survival[contributing], axis=1) <= 1e-15)

    profile = np.array([[np.nan]])
    prediction = predict_random_survival_forest(
        fit, profiles=profile, na_action="impute", random_state=12
    )
    assert np.isnan(profile[0, 0])
    assert prediction.survival.shape == (1, fit.time_grid.size)
    np.testing.assert_array_equal(prediction.row_indices, [0])
    repeated = predict_random_survival_forest(
        fit, profiles=profile, na_action="impute", random_state=12
    )
    np.testing.assert_array_equal(prediction.survival, repeated.survival)
    with pytest.raises(ValueError, match="prediction na_action"):
        predict_random_survival_forest(fit, profiles=profile, random_state=12)

    predictor_only = fit_random_survival_forest(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        x,
        na_action="impute",
        n_trees=4,
        mtry=1,
        nodesize=1,
        nsplit=0,
        sample_fraction=0.8,
        random_state=9,
        compute_oob=True,
    )
    assert predictor_only.oob is not None and predictor_only.oob.concordance_available


def test_complete_impute_mode_preserves_seeded_tree_and_oob_results() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    x = np.arange(6.0)[:, None]
    kwargs = dict(
        n_trees=4,
        mtry=1,
        nodesize=1,
        nsplit=0,
        replace=False,
        sample_fraction=0.8,
        random_state=2718,
        compute_oob=True,
    )
    ordinary = fit_random_survival_forest(time, event, x, **kwargs)
    impute = fit_random_survival_forest(time, event, x, na_action="impute", **kwargs)
    assert not impute.imputation_performed
    assert all(
        np.array_equal(left.feature, right.feature)
        and np.array_equal(left.threshold, right.threshold, equal_nan=True)
        and np.array_equal(left.left, right.left)
        for left, right in zip(ordinary.trees, impute.trees, strict=True)
    )
    np.testing.assert_array_equal(ordinary.oob.survival, impute.oob.survival)
    assert all(tree.imputation_donors is not None for tree in impute.trees)


def test_imputation_preflight_and_source_skipped_bootstrap_counts() -> None:
    with pytest.raises(ValueError, match="max_imputation_cells"):
        fit_random_survival_forest(
            [1.0, 2.0, np.nan, 4.0],
            [1.0, 0.0, 1.0, 0.0],
            [[1.0], [2.0], [3.0], [4.0]],
            na_action="impute",
            n_trees=2,
            random_state=3,
            max_imputation_cells=1,
        )
    fit = fit_random_survival_forest(
        [np.nan, 1.0, 2.0, 3.0],
        [1.0, 1.0, 0.0, 1.0],
        na_action="impute",
        n_trees=10,
        replace=True,
        sample_fraction=0.25,
        nodesize=1,
        random_state=7,
    )
    assert fit.requested_trees == 10
    assert fit.n_trees == 8
    assert fit.sampled_rows == 8


def test_prediction_omission_matches_complete_profiles_and_retains_original_positions() -> None:
    fit = fit_random_survival_forest(
        [1, 2, 3, 4],
        [1, 0, 1, 0],
        [[0], [1], [2], [3]],
        n_trees=3,
        nodesize=1,
        random_state=17,
    )
    profiles = np.array([[np.nan], [0.5], [np.nan], [2.5]])
    omitted = predict_random_survival_forest(fit, profiles=profiles, na_action="omit")
    complete = predict_random_survival_forest(fit, profiles=profiles[[1, 3]])
    np.testing.assert_array_equal(omitted.row_indices, [1, 3])
    np.testing.assert_array_equal(omitted.survival, complete.survival)
    np.testing.assert_array_equal(omitted.profile, complete.profile)
    assert np.isnan(profiles[[0, 2]]).all()
    with pytest.raises(ValueError, match="every prediction profile"):
        predict_random_survival_forest(fit, profiles=[[np.nan]], na_action="omit")
