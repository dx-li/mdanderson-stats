import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from mdanderson_stats import (
    fit_synergy_parametric,
    predict_synergy_parametric,
    synergy_parametric_response,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name):
    with (FIXTURES / f"synergy-parametric-{name}.csv").open(newline="") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize(
    "model,variant",
    [("greco", 1), ("greco", 2), ("greco", 3), ("machado", 1), ("plummer", 1), ("carter", 1)],
)
def test_original_author_kernels(model, variant):
    rows = [
        row for row in _rows("kernels") if row["model"] == model and int(row["variant"]) == variant
    ]
    count = 4 if model == "carter" else 5
    parameters = [float(rows[0][f"p{i}"]) for i in range(1, count + 1)]
    result = synergy_parametric_response(
        model,
        parameters,
        [float(row["dose1"]) for row in rows],
        [float(row["dose2"]) for row in rows],
    )
    np.testing.assert_allclose(
        result, [float(row["response"]) for row in rows], rtol=2e-12, atol=3e-14
    )


@pytest.mark.parametrize(
    "model,case",
    [
        (model, case)
        for model in ["greco", "machado", "plummer", "carter"]
        for case in ["paper", "synthetic"]
    ]
    + [("greco", "antagonistic")],
)
def test_original_author_nls_fits_and_uncertainty(model, case):
    rows = [row for row in _rows("inputs") if row["model"] == model and row["case"] == case]
    parameters = [row for row in _rows("fits") if row["model"] == model and row["case"] == case]
    fit = fit_synergy_parametric(
        [float(row["drug1"]) for row in rows],
        [float(row["drug2"]) for row in rows],
        [float(row["response"]) for row in rows],
        model=model,
    )
    assert fit.parameter_names == tuple(row["parameter"] for row in parameters)
    np.testing.assert_allclose(
        fit.initial_parameters,
        [float(row["initial"]) for row in parameters],
        rtol=1e-12,
        atol=1e-13,
    )
    # Original R nls uses its default 1e-5 convergence tolerance; Python refines further.
    np.testing.assert_allclose(
        fit.parameters, [float(row["estimate"]) for row in parameters], rtol=3e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        fit.standard_errors,
        [float(row["standard_error"]) for row in parameters],
        rtol=3e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit.t_statistics, [float(row["t_statistic"]) for row in parameters], rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        fit.p_values, [float(row["p_value"]) for row in parameters], rtol=1e-3, atol=1e-14
    )
    reference = [
        row for row in _rows("covariance") if row["model"] == model and row["case"] == case
    ]
    cov = np.array([float(row["covariance"]) for row in reference]).reshape(len(parameters), -1)
    cor = np.array([float(row["correlation"]) for row in reference]).reshape(len(parameters), -1)
    np.testing.assert_allclose(fit.covariance, cov, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(fit.parameter_correlation, cor, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(fit.residual_sum_squares, float(parameters[0]["sse"]), rtol=1e-9)
    np.testing.assert_allclose(
        fit.residual_standard_error, float(parameters[0]["residual_standard_error"]), rtol=1e-9
    )
    assert fit.residual_degrees_freedom == int(parameters[0]["df"])
    assert 1 <= fit.evaluations <= 1000
    assert not fit.parameters.flags.writeable
    assert not fit.covariance.flags.writeable
    np.testing.assert_allclose(
        predict_synergy_parametric(fit, fit.dose1, fit.dose2), fit.fitted_response
    )


def test_closed_forms_detect_greco_product_dose_interaction_and_equal_slopes():
    d1, d2 = np.array([0.2, 1.0, 3.0]), np.array([0.4, 0.8, 2.0])
    a, b, m, alpha = d1 / 1.8, d2 / 0.9, -0.7, 0.6
    greco = synergy_parametric_response("greco", [m, m, 1.8, 0.9, alpha], d1, d2)
    np.testing.assert_allclose(greco, expit(m * np.log(a + b + alpha * a * b)), rtol=2e-14)
    eta = 0.6
    machado = synergy_parametric_response("machado", [m, m, 1.8, 0.9, eta], d1, d2)
    np.testing.assert_allclose(machado, expit((m / eta) * np.log(a**eta + b**eta)), rtol=2e-14)
    plummer = synergy_parametric_response("plummer", [0.3, -0.8, 0.2, 0.0, 0.6], d1, d2)
    effective = d1 + np.exp(0.2) * d2 + 0.6 * np.sqrt(d1 * np.exp(0.2) * d2)
    np.testing.assert_allclose(plummer, expit(0.3 - 0.8 * np.log(effective)), rtol=2e-14)


@pytest.mark.parametrize(
    "model,parameters,interaction_index,additive,antagonistic",
    [
        ("greco", [-0.7, -1.3, 1.8, 0.9, 0.2], 4, 0.0, -0.03),
        ("machado", [-0.7, -1.3, 1.8, 0.9, 0.6], 4, 1.0, 1.5),
        ("plummer", [0.3, -0.8, 0.2, 0.4, 0.6], 4, 0.0, -0.3),
        ("carter", [0.8, -0.4, -0.6, -0.08], 3, 0.0, 0.08),
    ],
)
def test_interaction_direction_is_for_fraction_surviving(
    model, parameters, interaction_index, additive, antagonistic
):
    synergy = float(synergy_parametric_response(model, parameters, 0.7, 0.9))
    null = parameters.copy()
    null[interaction_index] = additive
    anti = parameters.copy()
    anti[interaction_index] = antagonistic
    assert synergy < float(synergy_parametric_response(model, null, 0.7, 0.9))
    assert float(synergy_parametric_response(model, anti, 0.7, 0.9)) > float(
        synergy_parametric_response(model, null, 0.7, 0.9)
    )


def test_control_mean_is_an_explicit_native_plot_convention():
    rows = [
        row for row in _rows("inputs") if row["model"] == "greco" and row["case"] == "synthetic"
    ]
    fit = fit_synergy_parametric(
        [float(r["drug1"]) for r in rows],
        [float(r["drug2"]) for r in rows],
        [float(r["response"]) for r in rows],
        model="greco",
    )
    assert not fit.included_rows[0]
    assert fit.observed_control_response == 0.95
    assert float(predict_synergy_parametric(fit, 0.0, 0.0)) == 0.95
    assert float(synergy_parametric_response("greco", fit.parameters, 0.0, 0.0)) == 1.0


def test_greco_rejects_ambiguous_negative_interaction_at_large_new_doses():
    with pytest.raises(ValueError, match="unique monotone"):
        synergy_parametric_response("greco", [-1.0, -1.0, 1.0, 1.0, -0.1], 100.0, 100.0)
    result = synergy_parametric_response("greco", [-1.0, -1.0, 1.0, 1.0, -0.1], 0.5, 0.5)
    assert 0 < result < 1


@pytest.mark.parametrize(
    "model,parameters,d1,d2",
    [
        ("unknown", [1, 2, 3, 4], 1.0, 1.0),
        ("carter", [1, 2, 3], 1.0, 1.0),
        ("carter", [1, 2, 3, 4], -1.0, 1.0),
        ("carter", [1, 2, 3, 4], 1j, 1.0),
        ("carter", [1, 2, 3, 4], np.ones((501, 1)), np.ones((1, 501))),
        ("greco", [0.7, -1.0, 1.0, 1.0, 0.0], 1.0, 1.0),
        ("machado", [-1.0, -1.0, 1.0, 1.0, 0.0], 1.0, 1.0),
        ("plummer", [0.3, -1.0, 0.0, -1.0, 0.0], 1.0, 1.0),
        ("carter", [1, float("nan"), 3, 4], 1.0, 1.0),
    ],
)
def test_prediction_validation(model, parameters, d1, d2):
    with pytest.raises(ValueError):
        synergy_parametric_response(model, parameters, d1, d2)


def test_training_validation_and_real_evaluation_budget():
    d1 = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 0, 0, 0, 0, 0, 0.2, 1.0, 4.0, 1.0])
    d2 = np.array([0, 0, 0, 0, 0, 0.1, 0.3, 1.0, 3.0, 10.0, 0.2, 1.0, 4.0, 2.0])
    response = synergy_parametric_response("greco", [-0.7, -1.3, 1.8, 0.9, 1.2], d1, d2)
    response = expit(np.log(response / (1 - response)) + 0.04 * np.cos(np.arange(len(d1)) * 1.7))
    with pytest.raises(ArithmeticError, match="evaluation budget"):
        fit_synergy_parametric(d1, d2, response, model="greco", max_evaluations=1)
    for invalid in [0, 2001, True, 1.5]:
        with pytest.raises(ValueError, match="max_evaluations"):
            fit_synergy_parametric(d1, d2, response, model="greco", max_evaluations=invalid)
    invalid = response.copy()
    invalid[0] = 1.0
    with pytest.raises(ValueError, match="noncontrol"):
        fit_synergy_parametric(d1, d2, invalid, model="greco")
    with pytest.raises(ValueError, match="two distinct"):
        fit_synergy_parametric(
            np.tile(d1[10:], 2), np.tile(d2[10:], 2), np.tile(response[10:], 2), model="carter"
        )
