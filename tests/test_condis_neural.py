import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.condis import condis_impute
from mdanderson_stats.condis_neural import (
    _predict_gradient,
    condis_neural_refine,
    fit_condis_neural,
)


def test_one_hidden_unit_matches_closed_form_forward_objective_and_gradient():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.0, np.log(2.0), -np.log(2.0), 1.0, 2.0])
    objective, gradient, prediction = _predict_gradient(x, y, weights, 1, 0.1)
    np.testing.assert_allclose(prediction, [2.0, 7.0 / 3.0, 5.0 / 3.0, 2.0], atol=1e-14)
    assert objective == pytest.approx(62.0 / 9.0 + 0.1 * (5.0 + 2.0 * np.log(2.0) ** 2))
    np.testing.assert_allclose(
        gradient,
        [-1.8888888888888888, -1.5650742675917146, -3.323814621297174, -3.8, -1.0444444444444443],
        atol=1e-13,
    )
    fit = fit_condis_neural(
        x, y, hidden_size=1, decay=0.1, initial_weights=weights, max_iterations=0
    )
    assert fit.status == "not_run"
    assert fit.convergence_code == 0
    np.testing.assert_array_equal(fit.weights, weights)
    optimized = fit_condis_neural(
        x, y, hidden_size=1, decay=0.1, initial_weights=weights, max_iterations=5
    )
    assert optimized.status == "iteration_limit"
    assert optimized.objective < fit.objective
    assert optimized.gradient_evaluations > 0


def test_refinement_tunes_nine_candidates_and_restores_observed_events():
    time = np.arange(1.0, 19.0)
    status = np.tile([1, 0], 9)
    imputation = condis_impute(time, status, horizon=18.0)
    covariates = np.column_stack((np.linspace(-1, 1, time.size), time % 4))
    folds = np.arange(time.size) % 3
    result = condis_neural_refine(
        imputation,
        covariates,
        folds=3,
        fold_ids=folds,
        random_state=4,
        max_iterations=0,
    )
    assert result.tuning_grid.shape == (9, 2)
    assert result.fold_rmse.shape == (3, 9)
    assert result.selected_hidden_size in (1, 3, 5)
    assert result.selected_decay in (0.0, 0.0001, 0.1)
    assert not result.fold_ids.flags.writeable
    assert not result.selected_fit.weights.flags.writeable
    np.testing.assert_array_equal(result.refined_time[status == 1], time[status == 1])


def test_optimizer_matches_native_short_trajectory_checkpoint():
    fixtures = Path(__file__).parent / "fixtures"
    model_inputs = json.loads((fixtures / "condis-models-inputs.json").read_text())["cases"][
        "ordinary"
    ]
    reference = json.loads((fixtures / "condis-nnet-native.json").read_text())["cases"]["ordinary"][
        "trajectory"
    ]
    design = np.column_stack((model_inputs["status"], model_inputs["covariates"]))
    checkpoint = next(fit for fit in reference["fits"] if fit["maxit"] == 10)
    fit = fit_condis_neural(
        design,
        model_inputs["imputed_time"],
        hidden_size=reference["size"],
        decay=reference["decay"],
        initial_weights=reference["initial_weights"],
        max_iterations=10,
    )
    assert fit.convergence_code == checkpoint["convergence"] == 1
    assert fit.objective == pytest.approx(checkpoint["objective"], rel=0, abs=2e-10)
    np.testing.assert_allclose(fit.weights, checkpoint["fitted_weights"], rtol=0, atol=2e-10)
