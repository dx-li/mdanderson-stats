"""Native AFT references and finite-fit boundaries."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.parametric_survival import (
    fit_parametric_survival,
    predict_parametric_survival,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name):
    with (FIXTURES / f"parametric-survival-{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_native_fit_joint_covariance_and_survival_references():
    data, metrics, predictions = _rows("input"), _rows("metric"), _rows("prediction")
    time = [float(row["time"]) for row in data]
    event = [int(row["event"]) for row in data]
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in data])
    for distribution in ("weibull", "lognormal", "loglogistic"):
        for design in ("covariates", "intercept"):
            case = f"{distribution}_{design}"
            fit = fit_parametric_survival(
                time, event, x if design == "covariates" else None, distribution=distribution
            )
            fields = {
                "parameters": np.r_[fit.coefficients, np.log(fit.sigma)],
                "covariance": fit.covariance,
                "information": fit.information,
                "log_likelihood": np.asarray(fit.log_likelihood),
            }
            for name, actual in fields.items():
                expected = [
                    float(row["value"])
                    for row in metrics
                    if row["case"] == case and row["metric"] == name
                ]
                np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-7, atol=2e-8)
            for profile in ("mean", "explicit"):
                for kind in ("native", "custom"):
                    rows = [
                        row
                        for row in predictions
                        if row["case"] == case
                        and row["profile"] == profile
                        and row["time_kind"] == kind
                    ]
                    times = np.unique([float(row["time"]) for row in rows])
                    profiles = np.array(
                        [[float(row["x1"]), float(row["x2"])] for row in rows[:: len(times)]]
                    )
                    result = predict_parametric_survival(
                        fit, times, profiles if design == "covariates" else np.empty((5, 0))
                    )
                    for name in ("log_survival", "survival", "lower", "upper"):
                        expected = np.array([float(row[name]) for row in rows]).reshape(5, -1)
                        np.testing.assert_allclose(
                            getattr(result, name), expected, rtol=2e-7, atol=2e-8
                        )


def test_censored_group_with_no_events_has_no_finite_location_estimate():
    for distribution in ("weibull", "lognormal", "loglogistic"):
        with pytest.raises(ValueError, match="monotone likelihood"):
            fit_parametric_survival(
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                distribution=distribution,
            )
