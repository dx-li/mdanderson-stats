import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats.condis import condis_impute
from mdanderson_stats.condis_forest import (
    condis_forest_refine,
    fit_condis_forest,
    predict_condis_forest,
)


def _fixtures() -> tuple[dict, dict]:
    root = Path(__file__).parent / "fixtures"
    return (
        json.loads((root / "condis-models-inputs.json").read_text()),
        json.loads((root / "condis-rf-native.json").read_text()),
    )


def test_explicit_uniform_replay_matches_first_three_native_trees() -> None:
    inputs, references = _fixtures()
    case = inputs["cases"]["ordinary"]
    trace = references["cases"]["ordinary"]["trace"]
    x = np.column_stack((case["status"], case["covariates"]))
    y = np.asarray(case["imputed_time"])
    fit = fit_condis_forest(
        x,
        y,
        n_trees=3,
        mtry=2,
        nodesize=5,
        uniform_draws=trace["uniform_draws"],
        keep_inbag=True,
    )

    assert [tree.status.size for tree in fit.trees] == [25, 29, 29]
    assert np.array_equal(fit.inbag_counts, np.asarray(trace["inbag_counts"])[:, :3])
    for tree_index, tree in enumerate(fit.trees):
        native = trace["forest"]
        size = native["ndbigtree"][tree_index]
        assert np.array_equal(tree.feature, np.asarray(native["bestvar"])[:size, tree_index] - 1)
        # randomForest's codes are -3 for interior and -1 for terminal nodes.
        expected_status = np.where(
            np.asarray(native["nodestatus"])[:size, tree_index] == -3, -1, -3
        )
        assert np.array_equal(tree.status, expected_status)
        assert np.array_equal(tree.threshold, np.asarray(native["xbestsplit"])[:size, tree_index])

    predictions = predict_condis_forest(fit, x, individual=True)
    assert np.max(np.abs(predictions - np.asarray(trace["individual"])[:, :3])) < 1e-14


def test_tiny_response_and_adjacent_float_split_remain_representable() -> None:
    lower = 1.0
    upper = np.nextafter(lower, np.inf)
    x = np.array([[lower], [lower], [lower], [upper], [upper], [upper]])
    y = 1e-200 * np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    bootstrap = np.arange(6, dtype=float) / 6.0
    # Feature selection and the two native equality paths follow bootstrap.
    fit = fit_condis_forest(
        x,
        y,
        n_trees=1,
        mtry=1,
        nodesize=5,
        uniform_draws=np.r_[bootstrap, 0.0, 0.0, 0.0],
    )
    prediction = predict_condis_forest(fit, x)
    assert np.isfinite(prediction).all()
    assert prediction[0] < prediction[-1]
    assert fit.trees[0].threshold[0] == lower

    constant = fit_condis_forest(x, np.full(6, 1e-200), n_trees=1, mtry=1)
    assert np.all(predict_condis_forest(constant, x) == 1e-200)


def test_condis_wrapper_includes_status_once_and_uses_covariate_mtry() -> None:
    imputation = condis_impute(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 0, 1, 0, 1, 0, 1, 0],
        horizon=8,
    )
    covariates = np.column_stack([np.arange(8), np.arange(8) % 3, np.arange(8) ** 2, np.ones(8)])
    folds = np.repeat(np.arange(4), 2)
    result = condis_forest_refine(
        imputation,
        covariates,
        folds=4,
        fold_ids=folds,
        n_trees=1,
        random_state=17,
    )
    assert result.mtry == 2  # round(sqrt(4 original covariates))
    assert result.fit.predictor_count == 5  # four covariates plus status
    assert result.fold_ids.dtype == np.int64
    assert np.all(np.isfinite(result.refined_time))

    for time_scale in (1e-200, 1e200):
        scaled_input = replace(
            imputation,
            observed_time=imputation.observed_time * time_scale,
            imputed_time=imputation.imputed_time * time_scale,
            remaining_time=imputation.remaining_time * time_scale,
            horizon=imputation.horizon * time_scale,
        )
        scaled = condis_forest_refine(
            scaled_input,
            covariates,
            folds=4,
            fold_ids=folds,
            n_trees=1,
            random_state=17,
        )
        np.testing.assert_allclose(
            scaled.mean_rmse / time_scale, result.mean_rmse, rtol=2e-13, atol=0
        )
