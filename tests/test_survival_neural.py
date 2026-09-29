import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.survival_neural import (
    SurvivalNeuralFit,
    _backprop,
    _cox_time_gradient,
    _deepsurv_gradient,
    _forward,
    fit_survival_neural,
    predict_survival_neural,
    survival_neural_contour,
)
from mdanderson_stats.survival_neural_discrete import _loss_gradient

FIXTURES = Path(__file__).parent / "fixtures"


def _array(value):
    return np.asarray(value, dtype=np.float64)


def test_all_five_loss_gradients_and_cox_predictions_match_decimal_oracle():
    for case in json.loads((FIXTURES / "survival-neural-cox.json").read_text()):
        time, event, x = _array(case["time"]), _array(case["event"]), _array(case["x"])
        weights = tuple(_array(layer) for layer in case["weights"])
        if case["family"] == "coxtime":
            loss, gradient, baseline_times, log_hazard = _cox_time_gradient(
                x,
                time,
                event,
                weights,
                float(case.get("time_center", 0.0)),
                float(case.get("time_scale", 1.0)),
            )
        else:
            activation, preactivation = _forward(x, weights)
            loss, output_gradient, baseline_times, log_hazard = _deepsurv_gradient(
                time, event, activation[-1]
            )
            gradient = _backprop(activation, preactivation, weights, output_gradient)
        assert_allclose(loss, float(case["loss"]), rtol=2e-14, atol=2e-14)
        for actual, expected in zip(gradient, case["gradient"], strict=True):
            assert_allclose(actual, _array(expected), rtol=2e-13, atol=2e-13)
        assert_allclose(baseline_times, _array(case["event_times"]), atol=0, rtol=0)
        assert_allclose(log_hazard, _array(case["baseline_log_hazard"]), rtol=2e-14, atol=2e-14)
        fit = SurvivalNeuralFit(
            case["family"],
            tuple(np.array(w, copy=True) for w in weights),
            np.ones(x.shape[1]),
            np.zeros(x.shape[1]),
            np.ones(x.shape[1]),
            float(case.get("time_center", 0.0)),
            float(case.get("time_scale", 1.0)),
            np.empty(0),
            baseline_times,
            log_hazard,
            np.empty(0),
            1,
            1,
            False,
            1,
            time.size,
        )
        predicted = predict_survival_neural(
            fit, _array(case["prediction_x"]), _array(case["prediction_times"])
        )
        assert_allclose(predicted, _array(case["prediction"]), rtol=2e-13, atol=2e-13)


def test_discrete_losses_and_logit_gradients_match_decimal_oracle():
    for case in json.loads((FIXTURES / "survival-neural-discrete.json").read_text()):
        loss, gradient, baseline_times, log_hazard = _loss_gradient(
            case["family"],
            case["time"],
            case["event"],
            case["logits"],
            case["cuts"],
            alpha=float(case["alpha"]),
            rank_sigma=float(case["rank_sigma"]),
        )
        assert_allclose(loss, float(case["loss"]), rtol=2e-14, atol=2e-14)
        assert_allclose(gradient, _array(case["gradient"]), rtol=2e-13, atol=2e-13)
        assert baseline_times.size == log_hazard.size == 0


def test_seeded_fit_and_survival_contour_cover_each_family():
    time = [1, 2, 2, 3, 4, 5, 6, 7]
    event = [1, 1, 0, 1, 1, 0, 1, 0]
    x = [[-1.0], [-0.4], [0.2], [0.5], [0.8], [1.0], [1.3], [1.5]]
    cuts = [0.0, 2.0, 4.0, 8.0]
    for family in ("coxtime", "deepsurv", "deephit", "loghaz", "pchazard"):
        cut_option = cuts if family in ("deephit", "loghaz", "pchazard") else None
        fit = fit_survival_neural(
            time,
            event,
            x,
            family=family,
            hidden_layers=(3,),
            max_epochs=3,
            patience=2,
            random_state=9,
            cuts=cut_option,
        )
        result = predict_survival_neural(fit, [[0.0], [1.0]], [0, 1, 3, 5, 8])
        assert result.shape == (2, 5)
        assert np.isfinite(result).all()
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert np.all(np.diff(result, axis=1) <= 1e-14)
        assert fit.weights[0].flags.writeable is False
        assert fit.epochs_run <= 3

    contour = survival_neural_contour(
        time,
        event,
        x,
        0,
        family="deepsurv",
        grid=[-0.5, 0.5],
        times=[1, 3],
        hidden_layers=(3,),
        max_epochs=2,
        patience=1,
        random_state=9,
    )
    assert contour.survival.shape == (2, 2)


def test_pchazard_rejects_unrepresentable_initial_event_and_prediction_preflights_work():
    with np.testing.assert_raises_regex(ValueError, "first cut"):
        fit_survival_neural(
            [0, 1, 2],
            [1, 1, 0],
            [[0], [1], [2]],
            family="pchazard",
            cuts=[0, 1, 2],
            hidden_layers=(2,),
            max_epochs=1,
            patience=1,
        )

    oversized = SurvivalNeuralFit(
        "coxtime",
        (np.zeros((3, 1)), np.zeros((2, 1))),
        np.ones(1),
        np.zeros(1),
        np.ones(1),
        0.0,
        1.0,
        np.empty(0),
        np.arange(100_000, dtype=np.float64),
        np.zeros(100_000),
        np.empty(0),
        1,
        1,
        False,
        1,
        3,
    )
    with np.testing.assert_raises_regex(ValueError, "work exceeds"):
        predict_survival_neural(oversized, np.zeros((2000, 1)), [1.0])


def test_pchazard_prediction_interpolates_integrated_interval_hazards():
    fit = SurvivalNeuralFit(
        "pchazard",
        (np.zeros((2, 2)), np.zeros((3, 2))),
        np.ones(1),
        np.zeros(1),
        np.ones(1),
        0.0,
        1.0,
        np.array([0.0, 1.0, 3.0]),
        np.empty(0),
        np.empty(0),
        np.empty(0),
        1,
        1,
        False,
        1,
        3,
    )
    prediction = predict_survival_neural(fit, [[0.0]], [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    expected = [[1.0, 2**-0.5, 0.5, 2**-1.5, 0.25, 0.25]]
    assert_allclose(prediction, expected, rtol=2e-15, atol=2e-15)
