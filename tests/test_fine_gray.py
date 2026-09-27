"""Regression tests against native cmprsk 2.2-12 numerical references."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.fine_gray import fine_gray, fine_gray_predict

FIXTURES = Path(__file__).parent / "fixtures"


def _csv(name):
    with (FIXTURES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _input():
    rows = _csv("fine-gray-input.csv")
    time = np.array([float(row["time"]) for row in rows])
    status = np.array([int(row["status"]) for row in rows])
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
    groups = np.array([row["group"] for row in rows])
    return time, status, x, groups


def _fit(case):
    time, status, x, groups = _input()
    kwargs = {}
    design = x
    shift = 0.0
    if case not in ("fixed_one", "zero_censor"):
        kwargs["censoring_groups"] = groups
    if case == "mixed_groups":
        kwargs.update(time_covariates=x[:, 0], time_functions=lambda t: (t / 10)[:, None])
    elif case == "time_only":
        design = None
        kwargs.update(
            time_covariates=x,
            time_functions=lambda t: np.column_stack((1 + t / 10, np.sqrt(1 + t / 10))),
        )
    elif case == "target_two":
        kwargs["failcode"] = 2
    elif case == "zero_censor":
        time[1:4] = 0
    return fine_gray(time, status, design, **kwargs), shift


def test_fine_gray_matches_native_fit_sandwich_baseline_residual_and_cif_fixtures():
    metric_rows = _csv("fine-gray-metric.csv")
    event_rows = _csv("fine-gray-event.csv")
    prediction_rows = _csv("fine-gray-prediction.csv")
    cases = ("fixed_one", "fixed_groups", "mixed_groups", "time_only", "target_two", "zero_censor")
    metric_fields = {
        "coef": "coefficients",
        "score": "score",
        "inf": "observed_information",
        "var": "covariance",
        "invinf": "observed_information",
        "loglik": "log_likelihood",
        "loglik.null": "null_log_likelihood",
    }
    for case in cases:
        fit, shift = _fit(case)
        for metric, field in metric_fields.items():
            expected = np.array(
                [
                    float(row["value"])
                    for row in metric_rows
                    if row["case"] == case and row["metric"] == metric
                ]
            )
            if metric == "invinf":
                actual = np.linalg.solve(
                    fit.observed_information, np.eye(fit.coefficients.size)
                ).ravel()
            else:
                actual = np.asarray(getattr(fit, field)).ravel()
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
        expected_events = [row for row in event_rows if row["case"] == case]
        np.testing.assert_allclose(
            fit.baseline_increments,
            [float(row["baseline_increment"]) for row in expected_events],
            rtol=2e-9,
            atol=2e-10,
        )
        for column in range(fit.coefficients.size):
            np.testing.assert_allclose(
                fit.score_residuals[:, column],
                [float(row[f"residual_{column + 1}"]) for row in expected_events],
                rtol=2e-9,
                atol=2e-10,
            )
        profiles = {}
        for row in prediction_rows:
            if row["case"] == case:
                profiles.setdefault((row["profile"], row["row"]), []).append(row)
        for rows in profiles.values():
            expected = np.array([float(row["incidence"]) for row in rows])
            times = np.array([float(row["time"]) + shift for row in rows])
            first = rows[0]
            x_profile = (
                np.array([float(first["x1"]), float(first["x2"])]) if case != "time_only" else None
            )
            tcov = None
            if case == "mixed_groups":
                tcov = np.array([float(first["x1"])])
            elif case == "time_only":
                tcov = np.array([float(first["x1"]), float(first["x2"])])
            prediction = fine_gray_predict(fit, x_profile, times=times, time_covariates=tcov)
            np.testing.assert_allclose(
                prediction.cumulative_incidence[0], expected, rtol=2e-9, atol=2e-10
            )
            np.testing.assert_allclose(
                prediction.subdistribution_survival[0],
                np.exp(-prediction.cumulative_subdistribution_hazard[0]),
                rtol=2e-14,
                atol=0,
            )


def test_exact_left_censoring_limit_at_zero_and_offset_invariance():
    fit, _ = _fit("zero_censor")
    np.testing.assert_array_equal(fit.censoring_survival_left[:4], np.ones(4))

    time, status, x, _ = _input()
    base = fine_gray(time, status, x)
    shifted = fine_gray(time, status, x + 1e8)
    np.testing.assert_allclose(shifted.coefficients, base.coefficients, rtol=2e-7, atol=2e-8)
    profile = np.array([0.25, -0.1])
    base_cif = fine_gray_predict(base, profile, times=[0, 3, 8, 16]).cumulative_incidence
    shifted_cif = fine_gray_predict(
        shifted, profile + 1e8, times=[0, 3, 8, 16]
    ).cumulative_incidence
    np.testing.assert_allclose(shifted_cif, base_cif, rtol=2e-8, atol=2e-9)


def test_monotone_likelihood_and_bounded_risk_work_fail_clearly():
    with pytest.raises(ValueError, match="monotone likelihood"):
        fine_gray([1, 2, 3, 4], [1, 1, 1, 0], [3, 2, 1, 0])
    n = 2240
    with pytest.raises(ValueError, match="bounded row-event"):
        fine_gray(np.arange(n), np.ones(n), np.linspace(-1, 1, n))
