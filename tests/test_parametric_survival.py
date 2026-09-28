"""Native AFT references and finite-fit boundaries."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.parametric_survival import (
    fit_parametric_survival,
    predict_parametric_survival,
)
from mdanderson_stats.parametric_survival_contour import parametric_survival_contour

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


def test_contour_memory_budget_is_checked_before_fitting():
    # This design has no finite fit, but its oversized output must be rejected
    # before fitting or allocating profile-by-time surfaces.
    with pytest.raises(ValueError, match="combined parametric contour output"):
        parametric_survival_contour(
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            0,
            n_grid=2000,
            times=np.linspace(0, 6, 500),
        )


def test_monte_carlo_contour_shares_draws_across_both_profile_surfaces():
    for distribution in ("weibull", "gengamma", "gengamma.orig", "spline_hazard"):
        if distribution.startswith("gengamma"):
            with (FIXTURES / "generalized-gamma-input.csv").open(newline="") as stream:
                rows = [
                    row for row in csv.DictReader(stream) if row["case"] == "positive_covariates"
                ]
        else:
            rows = _rows("input")
        time = np.array([float(row["time"]) for row in rows])
        event = np.array([int(row["event"]) for row in rows])
        x = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
        result = parametric_survival_contour(
            time,
            event,
            x,
            0,
            distribution=distribution,
            grid=[-1.0, 0.0, 1.0],
            times=[0.0, 1.0, 10.0],
            interval_method="monte_carlo",
            draws=16,
            rng=19,
        )
        mc = result.monte_carlo
        assert mc is not None
        assert result.interval_method == "monte_carlo"
        assert mc.profile.shape[0] == result.grid.size + result.quantile_probabilities.size
        assert mc.parameter_draws.shape == (16, result.fit.scaled_parameters.size)
        assert np.shares_memory(result.lower, mc.lower)
        assert np.shares_memory(result.quantile_upper, mc.upper)
        assert np.array_equal(result.lower, mc.lower[: result.grid.size])
        assert np.array_equal(result.quantile_upper, mc.upper[result.grid.size :])
        assert result.se_log_cumulative_hazard.shape == result.lower.shape
        assert not result.lower.flags.writeable
        if distribution == "weibull":
            replay = parametric_survival_contour(
                time,
                event,
                x,
                0,
                distribution=distribution,
                grid=[-1.0, 0.0, 1.0],
                times=[0.0, 1.0, 10.0],
                interval_method="monte_carlo",
                parameter_draws=mc.parameter_draws,
                draws=2,
                rng=987,
            )
            assert replay.monte_carlo is not None
            assert np.array_equal(replay.lower, result.lower)
            assert np.array_equal(replay.quantile_upper, result.quantile_upper)


def test_monte_carlo_contour_work_is_rejected_before_model_fit():
    with pytest.raises(ValueError, match="combined Monte Carlo contour"):
        parametric_survival_contour(
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            0,
            grid=[0.0, 1.0],
            times=np.arange(1.0, 101.0),
            interval_method="monte_carlo",
            draws=100_000,
        )
