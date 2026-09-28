from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.interval_survival import (
    _maximal_intersections,
    fit_interval_survival,
    predict_interval_survival,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_rows(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / f"interval-survival-{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _case_arrays(
    case: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    rows = [row for row in _fixture_rows("inputs") if row["case"] == case]
    lower = np.array([float(row["lower"]) for row in rows])
    upper = np.array(
        [float(row["upper"]) if row["upper"].lower() != "inf" else np.inf for row in rows]
    )
    covariates = (
        np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
        if int(rows[0]["covariates"])
        else None
    )
    weights = np.array([float(row["weight"]) for row in rows])
    return (
        lower,
        upper,
        covariates,
        weights,
        np.array([[int(row["left_index"]), int(row["right_index"])] for row in rows]),
    )


def test_native_support_index_fit_mass_and_curve_references() -> None:
    metric_rows = _fixture_rows("metrics")
    support_rows = _fixture_rows("supports")
    curve_rows = _fixture_rows("curves")
    for case in ("mixed", "mixed_weighted", "turnbull", "current_status", "exact_right"):
        lower, upper, x, weights, expected_indices = _case_arrays(case)
        support_lower, support_upper, _, (left_index, right_index) = _maximal_intersections(
            lower, upper
        )
        np.testing.assert_array_equal(np.column_stack((left_index, right_index)), expected_indices)
        expected_support = [row for row in support_rows if row["case"] == case]
        np.testing.assert_allclose(
            support_lower,
            [float(row["original_lower"]) for row in expected_support],
            rtol=0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            support_upper,
            [float(row["upper"]) if row["upper"] != "Inf" else np.inf for row in expected_support],
            rtol=0,
            atol=1e-14,
        )

        fit = fit_interval_survival(lower, upper, x, weights=weights)
        expected_metrics = {
            row["metric"]: float(row["value"]) for row in metric_rows if row["case"] == case
        }
        assert fit.log_likelihood == pytest.approx(expected_metrics["log_likelihood"], abs=5e-9)
        assert fit.converged
        assert fit.score_error < 1e-7
        expected_mass = [float(row["mass"]) for row in expected_support]
        np.testing.assert_allclose(fit.support_mass, expected_mass, atol=3e-7, rtol=3e-7)
        if x is not None:
            expected_beta = np.array([expected_metrics["beta1"], expected_metrics["beta2"]])
            np.testing.assert_allclose(fit.coefficients, expected_beta, atol=2e-7, rtol=2e-7)

        case_curves = [row for row in curve_rows if row["case"] == case]
        profiles = (
            np.unique(
                np.array([[float(row["x1"]), float(row["x2"])] for row in case_curves]), axis=0
            )
            if x is not None
            else np.empty((1, 0))
        )
        times = np.r_[
            0.0,
            np.array([float(row["upper"]) for row in expected_support if row["upper"] != "Inf"]),
        ]
        prediction = predict_interval_survival(fit, times, profiles)
        for profile_index, profile in enumerate(profiles):
            curve = (
                [
                    row
                    for row in case_curves
                    if float(row["x1"]) == profile[0] and float(row["x2"]) == profile[1]
                ]
                if x is not None
                else case_curves
            )
            expected = np.array([float(row["survival"]) for row in curve[: times.size]])
            np.testing.assert_allclose(
                prediction.survival_lower[profile_index], expected, atol=4e-7, rtol=4e-7
            )
            np.testing.assert_allclose(
                prediction.survival_upper[profile_index], expected, atol=4e-7, rtol=4e-7
            )


def test_turnbull_identification_bounds_and_time_unit_invariance() -> None:
    lower = np.array([0.0, 1.0, 0.0, 2.0])
    upper = np.array([1.0, 2.0, 2.0, np.inf])
    expected_mass = np.array([3 / 8, 3 / 8, 1 / 4])
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    expected_lower = np.array([[1.0, 5 / 8, 5 / 8, 1 / 4, 1 / 4, 0.0]])
    expected_upper = np.array([[1.0, 1.0, 5 / 8, 5 / 8, 1 / 4, 1 / 4]])
    for factor in (1.0, 1e-100, 1e100):
        fit = fit_interval_survival(lower * factor, upper * factor)
        np.testing.assert_allclose(fit.support_mass, expected_mass, atol=2e-7)
        prediction = predict_interval_survival(fit, times * factor)
        np.testing.assert_allclose(prediction.survival_lower, expected_lower, atol=2e-7)
        np.testing.assert_allclose(prediction.survival_upper, expected_upper, atol=2e-7)
    tail = predict_interval_survival(fit, [1e-200, 1e200])
    np.testing.assert_allclose(tail.survival_lower, [[5 / 8, 0.0]], atol=2e-7)
    np.testing.assert_allclose(tail.survival_upper, [[1.0, 0.25]], atol=2e-7)


def test_global_weight_scaling_and_input_failures() -> None:
    lower = np.array([0.0, 1.0, 0.0, 2.0])
    upper = np.array([1.0, 2.0, 2.0, np.inf])
    fit = fit_interval_survival(lower, upper, weights=np.full(4, 1e-200))
    assert fit.log_likelihood == pytest.approx(-3.635634939595159e-200, rel=1e-7)
    with pytest.raises(ValueError, match="lower"):
        fit_interval_survival([2.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        fit_interval_survival(lower, upper, weights=[1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="all observations are right-censored"):
        fit_interval_survival([1.0, 2.0], [np.inf, np.inf])
