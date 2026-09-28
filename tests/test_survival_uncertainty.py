"""Shared-draw checks for source-compatible Monte Carlo survival intervals."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.generalized_gamma import GeneralizedGammaFit
from mdanderson_stats.parametric_survival import (
    ParametricSurvivalFit,
    predict_parametric_survival,
)
from mdanderson_stats.survival_spline import (
    SurvivalSplineFit,
    predict_survival_spline,
)
from mdanderson_stats.survival_uncertainty import predict_parametric_survival_mc

_FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _matrix(rows: list[dict[str, str]], case: str, metric: str) -> np.ndarray:
    selected = [row for row in rows if row["case"] == case and row["metric"] == metric]
    result = np.empty(
        (max(int(row["row"]) for row in selected), max(int(row["column"]) for row in selected))
    )
    for row in selected:
        result[int(row["row"]) - 1, int(row["column"]) - 1] = float(row["value"])
    return result


def _fit(case: str, distribution: str, metrics: list[dict[str, str]]):
    family = "generalized-gamma" if distribution in ("prentice", "stacy") else "parametric-survival"
    metric_case = "original_covariates" if distribution == "stacy" else case
    parameters = _matrix(metrics, metric_case, "parameters").reshape(-1)
    covariance = _matrix(metrics, metric_case, "covariance")
    information = np.linalg.inv(covariance)
    common = dict(
        covariance=covariance,
        information=information,
        log_likelihood=0.0,
        score_error=0.0,
        iterations=0,
        scaled_parameters=parameters,
        scaled_covariance=covariance,
        covariate_mean=np.zeros(2),
        covariate_scale=np.ones(2),
        log_time_center=0.0,
        log_time_scale=1.0,
    )
    if family == "parametric-survival":
        return ParametricSurvivalFit(
            distribution=distribution,
            sigma=float(np.exp(parameters[-1])),
            coefficients=parameters[:-1],
            **common,
        )
    if distribution == "prentice":
        return GeneralizedGammaFit(
            parameterization="prentice",
            coefficients=parameters[:-2],
            sigma=float(np.exp(parameters[-2])),
            q=float(parameters[-1]),
            shape=None,
            k=None,
            parameter_names=(),
            **common,
        )
    return GeneralizedGammaFit(
        parameterization="stacy",
        coefficients=parameters[:-2],
        sigma=None,
        q=None,
        shape=float(np.exp(parameters[-2])),
        k=float(np.exp(parameters[-1])),
        parameter_names=(),
        **common,
    )


def test_shared_draw_quantiles_match_base_r_for_all_parameterizations() -> None:
    draw_rows = _rows("survival-uncertainty-draws.csv")
    summary_rows = _rows("survival-uncertainty-summary.csv")
    metrics = _rows("generalized-gamma-metric.csv") + _rows("parametric-survival-metric.csv")
    profiles = np.array([[-0.7, 0.2], [0.4, -0.3], [1.2, 0.7]])
    times = np.array([0, 0.01, 0.2, 0.7, 1, 2, 4, 10, 100, 1e8])

    for case in sorted({row["case"] for row in draw_rows}):
        selected = [row for row in draw_rows if row["case"] == case]
        distribution = selected[0]["distribution"]
        columns = (
            ("beta0", "beta1", "beta2", "log_sigma")
            if distribution in ("weibull", "lognormal", "loglogistic")
            else ("beta0", "beta1", "beta2", "log_sigma", "Q")
            if distribution == "prentice"
            else ("beta0", "beta1", "beta2", "log_shape", "log_k")
        )
        supplied = np.array([[float(row[name]) for name in columns] for row in selected])
        fit = _fit(case, distribution, metrics)
        prediction = predict_parametric_survival_mc(
            fit, times, profiles, parameter_draws=supplied, rng=71
        )
        expected = [row for row in summary_rows if row["case"] == case]
        assert len(expected) == profiles.shape[0] * times.size
        for key, actual in (
            ("sd", prediction.simulated_sd),
            ("lower", prediction.lower),
            ("upper", prediction.upper),
        ):
            reference = np.array([float(row[key]) for row in expected]).reshape(actual.shape)
            assert np.allclose(actual, reference, rtol=2e-9, atol=2e-14), (case, key)
        assert np.array_equal(prediction.valid_draws, np.full(prediction.survival.shape, 64))


def test_invalid_draw_is_omitted_even_at_time_endpoints() -> None:
    fit = ParametricSurvivalFit(
        distribution="weibull",
        sigma=1.0,
        coefficients=np.array([0.0]),
        covariance=np.eye(2),
        information=np.eye(2),
        log_likelihood=0.0,
        score_error=0.0,
        iterations=0,
        scaled_parameters=np.array([0.0, 0.0]),
        scaled_covariance=np.eye(2),
        covariate_mean=np.empty(0),
        covariate_scale=np.empty(0),
        log_time_center=0.0,
        log_time_scale=1.0,
    )
    supplied = np.array([[0.0, 0.0], [0.0, 1000.0]])
    prediction = predict_parametric_survival_mc(fit, [0.0, 1.0, np.inf], parameter_draws=supplied)
    assert np.array_equal(prediction.valid_draws, [[1, 1, 1]])
    assert np.array_equal(prediction.survival, [[1.0, np.exp(-1.0), 0.0]])


def test_prentice_right_tail_keeps_zero_survival_finite() -> None:
    fit = GeneralizedGammaFit(
        parameterization="prentice",
        coefficients=np.array([-1000.0]),
        sigma=1.0,
        q=1.0,
        shape=None,
        k=None,
        parameter_names=(),
        covariance=np.eye(3),
        information=np.eye(3),
        log_likelihood=0.0,
        score_error=0.0,
        iterations=0,
        scaled_parameters=np.array([-1000.0, 0.0, 1.0]),
        scaled_covariance=np.eye(3),
        covariate_mean=np.empty(0),
        covariate_scale=np.empty(0),
        log_time_center=0.0,
        log_time_scale=1.0,
    )
    prediction = predict_parametric_survival_mc(
        fit, [1.0], parameter_draws=np.tile(fit.scaled_parameters, (2, 1))
    )
    assert np.array_equal(prediction.survival, [[0.0]])
    assert np.array_equal(prediction.valid_draws, [[2]])


def test_spline_shared_draw_intervals_match_base_r_and_keep_nonmonotone_draws() -> None:
    draw_rows = _rows("survival-spline-mc-draws.csv")
    summary_rows = _rows("survival-spline-mc-summary.csv")
    metric_rows = _rows("survival-spline-metric.csv")
    knot_rows = _rows("survival-spline-knots.csv")
    profiles = np.array([[-0.7, 0.2], [0.4, -0.3], [1.2, 0.7]])
    times = np.array([0, 0.03, 0.2, 0.7, 1, 2, 4, 10, 100, np.inf])

    for case in sorted({row["case"] for row in draw_rows}):
        scale, knot_text = case.split("_")
        k = int(knot_text)
        model_parameters = _matrix(metric_rows, case, "parameters").reshape(-1)
        covariance = _matrix(metric_rows, case, "covariance")
        knots = np.array([float(row["value"]) for row in knot_rows if row["case"] == case])
        fit = SurvivalSplineFit(
            scale=scale,
            k=k,
            knots=knots,
            coefficients=model_parameters,
            parameter_names=tuple(f"p{i}" for i in range(model_parameters.size)),
            covariance=covariance,
            information=np.eye(model_parameters.size),
            log_likelihood=0.0,
            score_error=0.0,
            iterations=0,
            scaled_parameters=model_parameters,
            scaled_covariance=covariance,
            covariate_mean=np.zeros(2),
            covariate_scale=np.ones(2),
            log_time_center=0.0,
            log_time_scale=1.0,
            scaled_knots=knots,
        )
        selected = [row for row in draw_rows if row["case"] == case]
        supplied = np.empty((64, model_parameters.size))
        for row in selected:
            supplied[int(row["draw"]) - 1, int(row["parameter"]) - 1] = float(row["value"])
        prediction = predict_parametric_survival_mc(fit, times, profiles, parameter_draws=supplied)
        expected = [row for row in summary_rows if row["case"] == case]
        for key, actual in (
            ("sd", prediction.simulated_sd),
            ("lower", prediction.lower),
            ("upper", prediction.upper),
        ):
            reference = np.array([float(row[key]) for row in expected]).reshape(actual.shape)
            assert np.allclose(actual, reference, rtol=2e-9, atol=2e-14), (case, key)
        assert np.array_equal(prediction.valid_draws, np.full(prediction.survival.shape, 64))
        assert prediction.spline_minimum_slope is not None
        if k == 4:
            assert prediction.spline_minimum_slope[1] < 0


def test_zero_knot_spline_links_reduce_to_existing_aft_models() -> None:
    times = np.array([0.1, 1.0, 10.0])
    profiles = np.array([[-0.4], [1.1]])
    gamma = np.array([-0.4, 1.7, 0.25])
    covariance = np.eye(gamma.size)
    standard = np.random.default_rng(1402).standard_normal((64, gamma.size))
    coefficient_draws = gamma + standard * np.array([0.08, 0.1, 0.06])
    coefficient_draws[:, 1] += 0.35
    for scale, distribution in (
        ("hazard", "weibull"),
        ("odds", "loglogistic"),
        ("normal", "lognormal"),
    ):
        spline_fit = SurvivalSplineFit(
            scale=scale,
            k=0,
            knots=np.array([-2.0, 2.0]),
            coefficients=gamma,
            parameter_names=("gamma0", "gamma1", "x1"),
            covariance=covariance,
            information=covariance,
            log_likelihood=0.0,
            score_error=0.0,
            iterations=0,
            scaled_parameters=gamma,
            scaled_covariance=covariance,
            covariate_mean=np.zeros(1),
            covariate_scale=np.ones(1),
            log_time_center=0.0,
            log_time_scale=1.0,
            scaled_knots=np.array([-2.0, 2.0]),
        )
        beta = np.array([-gamma[0] / gamma[1], -gamma[2] / gamma[1]])
        log_sigma = -np.log(gamma[1])
        aft_draws = np.column_stack(
            (
                -coefficient_draws[:, 0] / coefficient_draws[:, 1],
                -coefficient_draws[:, 2] / coefficient_draws[:, 1],
                -np.log(coefficient_draws[:, 1]),
            )
        )
        aft_fit = ParametricSurvivalFit(
            distribution=distribution,
            sigma=float(np.exp(log_sigma)),
            coefficients=beta,
            covariance=covariance,
            information=covariance,
            log_likelihood=0.0,
            score_error=0.0,
            iterations=0,
            scaled_parameters=np.r_[beta, log_sigma],
            scaled_covariance=covariance,
            covariate_mean=np.zeros(1),
            covariate_scale=np.ones(1),
            log_time_center=0.0,
            log_time_scale=1.0,
        )
        spline_prediction = predict_survival_spline(spline_fit, times, profiles)
        aft_prediction = predict_parametric_survival(aft_fit, times, profiles)
        assert np.allclose(spline_prediction.survival, aft_prediction.survival, rtol=2e-13)
        spline_mc = predict_parametric_survival_mc(
            spline_fit, times, profiles, parameter_draws=coefficient_draws
        )
        aft_mc = predict_parametric_survival_mc(aft_fit, times, profiles, parameter_draws=aft_draws)
        for name in ("survival", "lower", "upper", "simulated_sd"):
            assert np.allclose(
                getattr(spline_mc, name), getattr(aft_mc, name), rtol=3e-12, atol=2e-14
            ), (scale, name)
