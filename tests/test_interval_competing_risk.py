import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats._cdflib import _freeze
from mdanderson_stats.interval_competing_risk import (
    IntervalCompetingRiskFit,
    _basis,
    _choose_knots,
    _gor_cif,
    _gor_log_cif,
    _score_covariance,
    _score_rows,
    fit_interval_competing_risk,
    predict_interval_competing_risk,
)

FIXTURES = Path(__file__).parent / "fixtures"
CASES = {"fine_gray": (0.0, 0.0), "odds": (1.0, 1.0), "mixed": (0.0, 1.0)}


def _read(name: str) -> list[dict[str, str]]:
    with (FIXTURES / f"interval-competing-risk-{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_native_parameter_likelihood_curves_and_score_covariance() -> None:
    inputs = _read("inputs")
    lower = np.array([float(row["v"]) for row in inputs])
    upper = np.array([np.inf if row["u"] == "Inf" else float(row["u"]) for row in inputs])
    event = np.array([int(row["event"]) for row in inputs])
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in inputs])
    endpoints = np.r_[lower, upper[event > 0]]
    knots, boundaries = _choose_knots(endpoints, 0.5)
    span = boundaries[1] - boundaries[0]
    normalized_knots = (knots - boundaries[0]) / span
    basis_lower = _basis((lower - boundaries[0]) / span, normalized_knots, np.array([0.0, 1.0]))
    upper_used = np.where(event > 0, upper, boundaries[1])
    basis_upper = _basis(
        (upper_used - boundaries[0]) / span, normalized_knots, np.array([0.0, 1.0])
    )
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    x_scaled = (x - mean) / scale
    parameters = _read("parameters")
    covariance_rows = _read("covariance")
    curve_rows = _read("curves")

    for case, alpha in CASES.items():
        native = np.array([float(row["value"]) for row in parameters if row["case"] == case])
        n_basis = basis_lower.shape[1]
        phi = native[: 2 * n_basis].reshape(2, n_basis)
        beta = native[2 * n_basis :].reshape(2, 2)
        scaled_parameters = np.r_[(phi + (beta @ mean)[:, None]).ravel(), (beta * scale).ravel()]
        log_likelihood, score, score_rows = _score_rows(
            scaled_parameters,
            basis_lower,
            basis_upper,
            x_scaled,
            lower,
            upper_used,
            event,
            np.asarray(alpha),
            need_rows=True,
        )
        assert score_rows is not None
        expected_ll = {
            "fine_gray": -277.683200096177,
            "odds": -277.0203199692201,
            "mixed": -278.03084011105767,
        }[case]
        assert log_likelihood == pytest.approx(expected_ll, abs=2e-12)
        if case == "mixed":
            numerical_score = np.empty(score.size)
            for j in range(score.size):
                step = 1e-6 * max(1.0, abs(scaled_parameters[j]))
                plus, minus = scaled_parameters.copy(), scaled_parameters.copy()
                plus[j] += step
                minus[j] -= step
                ll_plus = _score_rows(
                    plus,
                    basis_lower,
                    basis_upper,
                    x_scaled,
                    lower,
                    upper_used,
                    event,
                    np.asarray(alpha),
                )[0]
                ll_minus = _score_rows(
                    minus,
                    basis_lower,
                    basis_upper,
                    x_scaled,
                    lower,
                    upper_used,
                    event,
                    np.asarray(alpha),
                )[0]
                numerical_score[j] = (ll_plus - ll_minus) / (2.0 * step)
            assert np.max(np.abs(numerical_score - score)) < 2e-5

        estimated_covariance = _score_covariance(score_rows, n_basis, 2)
        estimated_covariance /= np.outer(np.tile(scale, 2), np.tile(scale, 2))
        expected_covariance = np.zeros((4, 4))
        for row in covariance_rows:
            if row["case"] == case:
                expected_covariance[int(row["row"]) - 1, int(row["column"]) - 1] = float(
                    row["value"]
                )
        assert (
            np.linalg.norm(estimated_covariance - expected_covariance)
            / np.linalg.norm(expected_covariance)
            < 1e-12
        )

        max_curve_error = 0.0
        for row in curve_rows:
            if row["case"] != case:
                continue
            time = float(row["t"])
            profile = np.array([float(row["x1"]), float(row["x2"])])
            basis = _basis(
                np.array([(time - boundaries[0]) / span]), normalized_knots, np.array([0.0, 1.0])
            )[0]
            eta = np.array([basis @ phi[j] + profile @ beta[j] for j in range(2)])
            prediction = np.array([_gor_cif(np.array([eta[j]]), alpha[j])[0][0] for j in range(2)])
            expected = np.array([float(row["cif1"]), float(row["cif2"])])
            max_curve_error = max(max_curve_error, float(np.max(np.abs(prediction - expected))))
        assert max_curve_error < 2e-14


def test_fit_constrained_model_and_prediction() -> None:
    inputs = _read("inputs")
    lower = np.array([float(row["v"]) for row in inputs])
    upper = np.array([np.inf if row["u"] == "Inf" else float(row["u"]) for row in inputs])
    event = np.array([int(row["event"]) for row in inputs])
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in inputs])
    fit = fit_interval_competing_risk(lower, upper, event, x, alpha=(0.0, 1.0), k=0.5)
    assert fit.converged
    assert fit.kkt_error < 2e-5
    assert np.max(fit.lower_boundary_cif) <= fit.boundary_cif_tolerance * (1 + 1e-6)
    assert fit.max_joint_cif <= 1.0
    prediction = predict_interval_competing_risk(fit, np.linspace(0.0, 2.276, 31), [[0, 0]])
    assert prediction.cif1.shape == prediction.cif2.shape == (1, 31)
    assert np.all(np.diff(prediction.cif1, axis=1) >= -1e-12)
    assert np.all(np.diff(prediction.cif2, axis=1) >= -1e-12)
    assert np.all(prediction.cif1 + prediction.cif2 <= 1.0 + 1e-8)


def test_link_retains_finite_left_tail_and_safe_right_tail() -> None:
    for alpha in (0.0, 1.0, 2.0):
        eta = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0])
        log_cif = _gor_log_cif(eta, alpha)
        cif, density_eta, _ = _gor_cif(eta, alpha)
        assert np.isfinite(log_cif).all()
        assert np.isfinite(cif).all()
        assert np.isfinite(density_eta).all()
        assert np.all(np.diff(cif) >= 0.0)
        assert log_cif[0] == pytest.approx(-1000.0)


def test_prediction_rejects_joint_probability_violation_inside_range() -> None:
    fit = IntervalCompetingRiskFit(
        coefficients=_freeze(np.array([1.0, -1.0])),
        covariance=_freeze(np.eye(2)),
        alpha=_freeze(np.array([2.0, 2.0])),
        baseline_coefficients=_freeze(np.tile(np.array([-20.0, -12.0, -4.0, np.log(2.0)]), (2, 1))),
        knots=_freeze(np.empty(0)),
        boundary_knots=_freeze(np.array([0.0, 1.0])),
        covariate_mean=_freeze(np.array([0.0])),
        covariate_scale=_freeze(np.array([1.0])),
        covariate_unit_scale=_freeze(np.array([1.0])),
        covariate_center_scaled=_freeze(np.array([0.0])),
        covariate_scale_scaled=_freeze(np.array([1.0])),
        covariate_min=_freeze(np.array([-np.log(16.0)])),
        covariate_max=_freeze(np.array([np.log(16.0)])),
        log_likelihood=0.0,
        iterations=0,
        score_error=0.0,
        kkt_error=0.0,
        lower_boundary_cif=_freeze(np.zeros(2)),
        max_joint_cif=0.0,
        boundary_cif_tolerance=1e-7,
        converged=True,
    )
    with pytest.raises(ValueError, match="joint-probability"):
        predict_interval_competing_risk(fit, [0.1], [[0.0]])
