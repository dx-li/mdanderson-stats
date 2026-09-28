import numpy as np
import pytest

from mdanderson_stats.condis import condis_impute
from mdanderson_stats.condis_boosting import (
    _fit_boosted_path,
    _predict_tree,
    condis_boosting_refine,
)


def test_boosting_path_consumes_one_uniform_per_row_and_stops_on_constant_response():
    n = 80
    x = np.column_stack((np.arange(n, dtype=np.float64), np.ones(n)))
    y = np.full(n, 7.5)
    uniforms = np.linspace(0.0, np.nextafter(1.0, 0.0), n * 3).reshape(n, 3)

    trees, fitted, initial = _fit_boosted_path(x, y, uniforms, n_trees=3, depth=3)

    assert len(trees) == 3
    assert all(len(tree.nodes) == 1 for tree in trees)
    assert initial[0] == pytest.approx(7.5)
    np.testing.assert_array_equal(fitted, y)


def test_boosting_split_handles_adjacent_floating_point_values():
    lower = 1.0
    upper = np.nextafter(lower, np.inf)
    x = np.r_[np.full(40, lower), np.full(40, upper)][:, None]
    y = np.r_[np.zeros(40), np.ones(40)]
    uniforms = np.full((80, 1), 0.999)
    uniforms[:20, 0] = 0.0
    uniforms[40:60, 0] = 0.0

    trees, _, _ = _fit_boosted_path(x, y, uniforms, n_trees=1, depth=1)

    root = trees[0].nodes[0]
    assert root.feature == 0
    assert lower < root.cut <= upper
    prediction = _predict_tree(trees[0], x)
    assert np.all(prediction[:40] < prediction[40:])


def test_boosting_refit_restores_events_and_respects_smallest_native_fold_size():
    n = 86
    time = np.arange(1.0, n + 1.0)
    status = np.tile(np.asarray([0, 1], dtype=np.int64), n // 2)
    covariates = np.sin(time / 9.0)[:, None]
    imputation = condis_impute(time, status)
    folds = np.arange(n, dtype=np.int64) % 2

    result = condis_boosting_refine(
        imputation,
        covariates,
        folds=2,
        fold_ids=folds,
        tree_grid=(1,),
        depth_grid=(1,),
        random_state=19,
    )

    assert result.best_tree_count == 1
    assert result.best_depth == 1
    assert result.fold_rmse.shape == (1, 2, 1)
    assert np.isfinite(result.fitted_time).all()
    np.testing.assert_array_equal(result.refined_time[status == 1], time[status == 1])
    small_time = np.arange(1.0, 85.0)
    small_status = np.tile(np.asarray([0, 1], dtype=np.int64), 42)
    small_imputation = condis_impute(small_time, small_status)
    with pytest.raises(ValueError, match="at least 43"):
        condis_boosting_refine(
            small_imputation,
            np.sin(small_time / 9.0)[:, None],
            folds=2,
            fold_ids=np.arange(84, dtype=np.int64) % 2,
            tree_grid=(1,),
            depth_grid=(1,),
            random_state=19,
        )
