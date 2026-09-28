"""Native-reference checks for exact generalized-gamma survival fits."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.generalized_gamma import (
    fit_generalized_gamma,
    predict_generalized_gamma,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _case_rows(rows: list[dict[str, str]], case: str) -> list[dict[str, str]]:
    return [row for row in rows if row["case"] == case]


def _metric_matrix(rows: list[dict[str, str]], case: str, metric: str) -> np.ndarray:
    selected = [row for row in rows if row["case"] == case and row["metric"] == metric]
    n_rows = max(int(row["row"]) for row in selected)
    n_columns = max(int(row["column"]) for row in selected)
    result = np.empty((n_rows, n_columns))
    for row in selected:
        result[int(row["row"]) - 1, int(row["column"]) - 1] = float(row["value"])
    return result


def _fit_fixture_case(rows: list[dict[str, str]], case: str, parameterization: str):
    times = np.array([float(row["time"]) for row in rows])
    events = np.array([float(row["event"]) for row in rows])
    covariates = (
        np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
        if case.endswith("covariates")
        else None
    )
    return fit_generalized_gamma(
        times,
        events,
        covariates,
        parameterization=parameterization,  # type: ignore[arg-type]
        tolerance=2e-7,
    )


def test_prentice_fits_and_predictions_match_native_references() -> None:
    input_rows = _read_fixture("generalized-gamma-input.csv")
    metric_rows = _read_fixture("generalized-gamma-metric.csv")
    prediction_rows = _read_fixture("generalized-gamma-prediction.csv")
    cases = (
        "positive_covariates",
        "positive_intercept",
        "negative_covariates",
        "negative_intercept",
    )

    for case in cases:
        fit = _fit_fixture_case(_case_rows(input_rows, case), case, "prentice")
        actual = {
            "parameters": np.r_[fit.coefficients, np.log(fit.sigma), fit.q],
            "covariance": fit.covariance,
            "information": fit.information,
            "log_likelihood": np.array([[fit.log_likelihood]]),
        }
        for metric, values in actual.items():
            expected = _metric_matrix(metric_rows, case, metric).reshape(np.shape(values))
            atol = 5e-5 if metric == "information" else 5e-7
            assert np.allclose(values, expected, rtol=1e-5, atol=atol), (case, metric)

        rows = _case_rows(prediction_rows, case)
        grid = np.array(sorted({float(row["time"]) for row in rows}))
        if case.endswith("covariates"):
            profiles = np.array(
                [[float(row["x1"]), float(row["x2"])] for row in rows[:: len(grid)]]
            )
        else:
            profiles = np.empty((3, 0))
        prediction = predict_generalized_gamma(fit, grid, profiles)
        observed = np.array([float(row["log_survival"]) for row in rows]).reshape(
            profiles.shape[0], grid.size
        )
        assert np.allclose(prediction.log_survival, observed, rtol=2e-6, atol=1e-8)
        assert np.all(prediction.survival[:, 0] == 1)
        assert np.all(prediction.lower[:, 0] == 1)
        assert np.all(prediction.upper[:, 0] == 1)


@pytest.mark.parametrize(
    ("case", "source_case"),
    (("original_covariates", "positive_covariates"), ("original_intercept", "positive_intercept")),
)
def test_stacy_coordinates_match_native_positive_q_transform(case: str, source_case: str) -> None:
    input_rows = _read_fixture("generalized-gamma-input.csv")
    metric_rows = _read_fixture("generalized-gamma-metric.csv")
    fit = _fit_fixture_case(_case_rows(input_rows, source_case), source_case, "stacy")
    actual = {
        "parameters": np.r_[fit.coefficients, np.log(fit.shape), np.log(fit.k)],
        "covariance": fit.covariance,
        "information": fit.information,
        "log_likelihood": np.array([[fit.log_likelihood]]),
    }
    for metric, values in actual.items():
        expected = _metric_matrix(metric_rows, case, metric).reshape(np.shape(values))
        # The pinned R reference transforms the Prentice observed covariance;
        # direct Stacy-coordinate Hessians differ slightly at finite precision.
        rtol = 2e-3 if metric in {"covariance", "information"} else 2e-6
        assert np.allclose(values, expected, rtol=rtol, atol=1e-6), (case, metric)


def test_prediction_surface_budget_rejects_excessive_allocation() -> None:
    rows = _case_rows(_read_fixture("generalized-gamma-input.csv"), "positive_intercept")
    fit = _fit_fixture_case(rows, "positive_intercept", "prentice")
    with pytest.raises(ValueError, match="2,000,000 cells"):
        predict_generalized_gamma(fit, np.ones(1000), np.empty((300, 0)))
