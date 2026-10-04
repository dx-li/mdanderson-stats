from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import fit_random_survival_forest


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    event = np.array([1.0, 1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0])
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0], [7.0], [8.0]])
    return time, event, x


def _fit(*, nimpute: int = 1, compute_oob: bool = False):
    time, event, x = _case()
    return fit_random_survival_forest(
        time,
        event,
        x,
        na_action="impute",
        nimpute=nimpute,
        n_trees=3,
        sample_fraction=0.875,
        replace=False,
        nodesize=1,
        nsplit=0,
        ntime=None,
        compute_oob=compute_oob,
        random_state=2718,
    )


def test_nimpute_one_preserves_single_pass_outputs_and_rng_path() -> None:
    time, event, x = _case()
    baseline = fit_random_survival_forest(
        time,
        event,
        x,
        na_action="impute",
        n_trees=3,
        sample_fraction=0.875,
        replace=False,
        nodesize=1,
        nsplit=0,
        ntime=None,
        compute_oob=True,
        random_state=2718,
    )
    explicit = _fit(nimpute=1, compute_oob=True)
    assert explicit.imputation_passes == 1
    assert explicit.requested_imputation_passes == 1
    assert explicit.completed_time is None
    np.testing.assert_array_equal(explicit.time_grid, baseline.time_grid)
    for actual, expected in zip(explicit.trees, baseline.trees, strict=True):
        np.testing.assert_array_equal(actual.feature, expected.feature)
        np.testing.assert_array_equal(actual.threshold, expected.threshold)
        np.testing.assert_array_equal(actual.event_time, expected.event_time)
    assert explicit.oob is not None and baseline.oob is not None
    np.testing.assert_array_equal(explicit.oob.contributor_count, baseline.oob.contributor_count)
    np.testing.assert_array_equal(explicit.oob.survival, baseline.oob.survival)


def test_iterated_fit_returns_completed_data_and_original_event_grid() -> None:
    fit = _fit(nimpute=2)
    assert fit.requested_imputation_passes == 2
    assert fit.imputation_passes == 2
    assert fit.completed_time is not None
    assert fit.completed_event is not None
    assert fit.completed_covariates is not None
    assert fit.imputation_fallback_used is not None
    np.testing.assert_array_equal(fit.time_grid, [1.0, 2.0, 5.0, 7.0])
    assert np.isfinite(fit.completed_time).all()
    assert set(np.unique(fit.completed_event)) <= {0.0, 1.0}
    assert np.isfinite(fit.completed_covariates).all()
    assert not fit.completed_time.flags.writeable
    assert not fit.completed_event.flags.writeable
    assert not fit.completed_covariates.flags.writeable
    assert not fit.imputation_fallback_used.flags.writeable
    assert fit.total_sampled_rows == 2 * 3 * 7
    assert fit.total_node_count is not None and fit.total_node_count >= fit.node_count
    assert fit.total_leaf_event_records is not None
    assert fit.total_split_work is not None
    assert fit.total_imputation_work > 0


def test_complete_impute_with_multiple_requested_passes_is_single_pass() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    x = np.arange(6.0).reshape(-1, 1)
    common = dict(
        na_action="impute",
        n_trees=4,
        sample_fraction=0.75,
        replace=False,
        nodesize=1,
        nsplit=0,
        compute_oob=True,
        random_state=77,
    )
    one = fit_random_survival_forest(time, event, x, nimpute=1, **common)
    many = fit_random_survival_forest(time, event, x, nimpute=5, **common)
    assert many.requested_imputation_passes == 5
    assert many.imputation_passes == 1
    assert many.total_sampled_rows == many.sampled_rows
    np.testing.assert_array_equal(many.time_grid, one.time_grid)
    np.testing.assert_array_equal(many.oob.survival, one.oob.survival)
    for actual, expected in zip(many.trees, one.trees, strict=True):
        np.testing.assert_array_equal(actual.feature, expected.feature)
        np.testing.assert_array_equal(actual.threshold, expected.threshold)
        np.testing.assert_array_equal(actual.event_time, expected.event_time)


def test_nimpute_requires_imputation_mode_and_cumulative_sample_budget() -> None:
    time, event, x = _case()
    with pytest.raises(ValueError, match="nimpute > 1 requires na_action='impute'"):
        fit_random_survival_forest(time, event, x, nimpute=2, n_trees=1)
    with pytest.raises(ValueError, match="max_sampled_rows"):
        fit_random_survival_forest(
            time,
            event,
            x,
            na_action="impute",
            nimpute=2,
            n_trees=3,
            sample_fraction=0.875,
            nodesize=1,
            nsplit=0,
            max_sampled_rows=41,
            random_state=2718,
        )
