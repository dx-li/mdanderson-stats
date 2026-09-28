import numpy as np
import pytest

from mdanderson_stats.condis import condis_impute
from mdanderson_stats.condis_regularized import (
    _knn_predict,
    condis_regularized_refine,
)


@pytest.mark.parametrize("method", ["ridge", "lasso", "knn"])
def test_regularized_refinement_uses_explicit_folds_and_restores_events(method):
    time = np.arange(1.0, 25.0)
    status = np.tile([1, 0, 1, 0], 6)
    imputation = condis_impute(time, status, horizon=24.0)
    covariates = np.column_stack((np.linspace(-2.0, 2.0, time.size), time % 5))
    folds = np.arange(time.size, dtype=np.int64) % 4
    grid = {"neighbor_grid": [1, 3, 5]} if method == "knn" else {"lambda_grid": [0.1, 1.0, 3.0]}

    result = condis_regularized_refine(
        imputation,
        covariates,
        method=method,
        folds=4,
        fold_ids=folds,
        **grid,
    )

    assert result.method == method
    np.testing.assert_array_equal(result.fold_ids[0], folds)
    assert result.fold_rmse.shape == (1, 4, 3)
    assert result.mean_rmse.shape == result.sd_rmse.shape == (3,)
    assert result.best_value in result.tuning_values
    assert np.isfinite(result.fitted_time).all()
    np.testing.assert_array_equal(
        result.refined_time[status == 1], time[status == 1]
    )


def test_default_fold_assignments_are_repeatable():
    time = np.arange(1.0, 19.0)
    status = np.tile([1, 0], 9)
    imputation = condis_impute(time, status, horizon=18.0)
    covariates = np.arange(time.size, dtype=float)[:, None]
    first = condis_regularized_refine(
        imputation, covariates, method="knn", folds=4, neighbor_grid=[1, 3], random_state=71
    )
    second = condis_regularized_refine(
        imputation, covariates, method="knn", folds=4, neighbor_grid=[1, 3], random_state=71
    )
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_array_equal(first.fold_rmse, second.fold_rmse)


@pytest.mark.parametrize(
    ("train", "response", "expected"),
    [
        ([1.0, np.sqrt(1.00001)], [10.0, 20.0], 10.0),
        ([1.0, 1.0, np.sqrt(0.99999)], [10.0, 20.0, 30.0], 20.0),
        ([np.sqrt(0.99999), 1.0, 1.0], [10.0, 20.0, 30.0], 10.0),
        ([0.0, 0.0, 1.0], [10.0, 20.0, 30.0], 15.0),
    ],
)
def test_knn_tie_insertion_matches_caret_kernel(train, response, expected):
    prediction = _knn_predict(
        np.asarray(train, dtype=float)[:, None],
        np.asarray(response, dtype=float),
        np.asarray([[0.0]]),
        1,
    )
    assert prediction[0] == pytest.approx(expected)
