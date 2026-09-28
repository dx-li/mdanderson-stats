"""Focused comparisons with the pinned flexsurv native spline reference."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np

from mdanderson_stats.survival_spline import (
    _basis,
    _link_terms,
    fit_survival_spline,
    predict_survival_spline,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / f"survival-spline-{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_native_basis_and_distribution_values() -> None:
    basis_rows = _rows("basis")
    grouped: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
    for row in basis_rows:
        grouped[(int(row["k"]), float(row["z"]))].append(row)
    # Basis fixture is independent of coefficients; evaluate each (k,z) once.
    for (knot_count, z), values in grouped.items():
        reference = sorted(values, key=lambda row: int(row["column"]))
        knots = np.array([float(row["knot"]) for row in reference])
        got = _basis(knots, np.array([z]))[0]
        got_derivative = _basis(knots, np.array([z]), derivative=1)[0]
        np.testing.assert_allclose(got, [float(row["basis"]) for row in reference], atol=2e-12)
        np.testing.assert_allclose(
            got_derivative, [float(row["derivative"]) for row in reference], atol=2e-12
        )

    for row in _rows("distribution"):
        eta = np.array([float(row["eta"])])
        terms = _link_terms(eta, cast(Literal["hazard", "odds", "normal"], row["scale"]))
        log_survival, log_density_factor = terms[:2]
        np.testing.assert_allclose(
            log_survival[0], float(row["log_survival"]), rtol=2e-9, atol=2e-10
        )
        log_density = (
            log_density_factor[0]
            + np.log(float(row["slope"]))
            - np.log(float(row["time"]))
        )
        np.testing.assert_allclose(log_density, float(row["log_density"]), rtol=2e-9, atol=2e-10)


def test_native_fits_joint_information_and_predictions() -> None:
    input_rows = _rows("input")
    time = np.array([float(row["time"]) for row in input_rows])
    event = np.array([float(row["event"]) for row in input_rows])
    covariates = np.array([[float(row["x1"]), float(row["x2"])] for row in input_rows])
    metrics: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in _rows("metric"):
        metrics[row["case"]][row["metric"]].append(row)
    prediction_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows("prediction"):
        prediction_rows[row["case"]].append(row)

    covariance_errors: list[float] = []
    for case, groups in metrics.items():
        scale, knot_label = case.rsplit("_", 1)
        knot_count = int(knot_label)
        fit = fit_survival_spline(
            time,
            event,
            covariates,
            scale=cast(Literal["hazard", "odds", "normal"], scale),
            k=knot_count,
        )
        assert fit.score_error < 1e-7
        for metric, actual in (
            ("parameters", fit.coefficients),
            ("information", fit.information),
            ("covariance", fit.covariance),
        ):
            rows = sorted(groups[metric], key=lambda row: (int(row["row"]), int(row["column"])))
            expected = np.array([float(row["value"]) for row in rows]).reshape(actual.shape)
            if metric == "covariance":
                covariance_errors.append(
                    float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                )
                np.testing.assert_allclose(actual, expected, rtol=1.2e-2, atol=3e-2)
            elif metric == "information":
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)

        rows = prediction_rows[case]
        profile_ids = sorted({int(row["row"]) for row in rows})
        profiles = np.array(
            [
                [
                    float(next(row for row in rows if int(row["row"]) == pid)[name])
                    for name in ("x1", "x2")
                ]
                for pid in profile_ids
            ]
        )
        times = np.array(sorted({float(row["time"]) for row in rows}))
        prediction = predict_survival_spline(fit, times, profiles)
        for row in rows:
            profile_index = profile_ids.index(int(row["row"]))
            time_index = int(np.searchsorted(times, float(row["time"])))
            log_survival = prediction.log_survival[profile_index, time_index]
            native_log_survival = float(row["log_survival"])
            scale_for_error = max(abs(native_log_survival), 1.0)
            assert abs(log_survival - native_log_survival) / scale_for_error < 2e-5
            assert (
                prediction.lower[profile_index, time_index]
                <= prediction.survival[profile_index, time_index]
            )
            assert (
                prediction.survival[profile_index, time_index]
                <= prediction.upper[profile_index, time_index]
            )
            assert np.isfinite(prediction.se_log_cumulative_hazard[profile_index, time_index])
        zero = predict_survival_spline(fit, [0.0], profiles)
        np.testing.assert_array_equal(zero.survival, np.ones_like(zero.survival))
        np.testing.assert_array_equal(zero.se_log_cumulative_hazard, np.zeros_like(zero.survival))

    assert max(covariance_errors) < 1.2e-2
