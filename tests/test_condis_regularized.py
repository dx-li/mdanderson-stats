import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import condis_impute, condis_regularized_refine
from mdanderson_stats.condis_regularized import _knn_predict


@pytest.mark.parametrize(
    ("case_name", "method", "rtol"),
    [
        ("ordinary", "ridge", 2e-7),
        ("ordinary", "lasso", 1e-8),
        ("ordinary", "knn", 2e-13),
        ("wide", "lasso", 2e-6),
    ],
)
def test_native_tuning_and_full_sample_refit(case_name, method, rtol):
    fixtures = Path(__file__).parent / "fixtures"
    inputs = json.loads((fixtures / "condis-models-inputs.json").read_text())["cases"][case_name]
    native = json.loads((fixtures / "condis-models-native.json").read_text())["cases"][case_name]
    expected = native["learners"][method]
    imputation = condis_impute(inputs["observed_time"], inputs["status"])
    np.testing.assert_allclose(imputation.imputed_time, inputs["imputed_time"], rtol=2e-14)
    scores = expected["fold_rmse"]
    means = expected["mean_rmse"]
    fitted = expected["fitted_time"]
    selected = expected["selected"]
    if method == "lasso":
        # Native default stopping error is visible in both sets of fold scores.
        # Use its same penalty path with tighter tolerance and saved KKT checks.
        filename = (
            "condis-lasso-converged.json"
            if case_name == "wide"
            else "condis-lasso-ordinary-converged.json"
        )
        tight = json.loads((fixtures / filename).read_text())
        scores, means = tight["fold_rmse"], tight["mean_rmse"]
        best = int(np.argmin(means))
        selected = tight["tuning"][best]
        fitted = np.asarray(tight["all_fitted"])[:, best]
    result = condis_regularized_refine(
        imputation,
        inputs["covariates"],
        method=method,
        folds=inputs["folds"],
        fold_ids=native["fold_ids"],
    )
    np.testing.assert_allclose(result.fold_rmse[0], scores, rtol=rtol, atol=2e-13)
    np.testing.assert_allclose(result.mean_rmse, means, rtol=rtol, atol=2e-13)
    np.testing.assert_allclose(result.fitted_time, fitted, rtol=rtol, atol=2e-13)
    assert result.best_value == selected


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
    np.testing.assert_array_equal(result.refined_time[status == 1], time[status == 1])


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
