from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.interval_survival_bootstrap import (
    _sample_covariance,
    bootstrap_interval_survival_coefficients,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / f"interval-survival-bootstrap-{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _current_status_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.array([0.0, 0.0, 1.0, 1.0])
    upper = np.array([1.0, 1.0, np.inf, np.inf])
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    weights = np.array([2.0, 1.0, 1.0, 1.0])
    return lower, upper, x, weights


def test_explicit_weighted_row_tape_retains_all_coefficient_draws() -> None:
    lower, upper, x, weights = _current_status_data()
    tape = np.array([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]])
    result = bootstrap_interval_survival_coefficients(
        lower,
        upper,
        x,
        weights=weights,
        replicates=2,
        resample_indices=tape,
    )

    assert result.resample_size == 5  # ceil(sum(original case weights))
    assert result.replicate_status == ("ok", "ok")
    assert result.successful_replicates == 2
    assert result.failed_replicates == 0
    assert result.coefficient_samples.shape == (2, 1)
    assert result.covariance_conditional_on_success
    expected_variance = np.var(result.coefficient_samples[:, 0], ddof=1)
    assert result.covariance[0, 0] == pytest.approx(expected_variance)
    assert result.standard_error[0] == pytest.approx(np.sqrt(expected_variance))
    assert not result.coefficient_samples.flags.writeable
    assert not result.covariance.flags.writeable


def test_failed_resample_is_retained_and_covariance_is_undefined() -> None:
    lower, upper, x, weights = _current_status_data()
    tape = np.array(
        [
            [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 0],  # only one covariate value; slope is not identified
        ]
    )
    result = bootstrap_interval_survival_coefficients(
        lower,
        upper,
        x,
        weights=weights,
        replicates=2,
        resample_indices=tape,
    )

    assert result.replicate_status == ("ok", "fit_failed")
    assert result.replicate_errors[0] is None
    assert result.replicate_errors[1]
    assert np.isnan(result.coefficient_samples[1]).all()
    assert result.successful_replicates == 1
    assert result.failed_replicates == 1
    assert np.isnan(result.covariance).all()
    assert np.isnan(result.standard_error).all()


def test_work_preflight_does_not_consume_generator_state() -> None:
    lower = np.tile([0.0, 0.0, 1.0, 1.0], 25)
    upper = np.tile([1.0, 1.0, np.inf, np.inf], 25)
    x = np.arange(lower.size, dtype=np.float64)[:, None]
    weights = np.ones(lower.size)
    rng = np.random.default_rng(91)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="worst-case bootstrap fitting work"):
        bootstrap_interval_survival_coefficients(
            lower,
            upper,
            x,
            weights=weights,
            replicates=5_000,
            max_iterations=5_000,
            rng=rng,
        )
    assert rng.bit_generator.state == before


def test_sampling_work_preflight_does_not_consume_generator_state() -> None:
    lower = np.array([0.0, 0.0, 1.0, 1.0])
    upper = np.array([1.0, 1.0, np.inf, np.inf])
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    weights = np.full(4, 10_000.0)
    rng = np.random.default_rng(17)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="sampling and row-collapse work"):
        bootstrap_interval_survival_coefficients(
            lower,
            upper,
            x,
            weights=weights,
            replicates=2_000,
            rng=rng,
        )
    assert rng.bit_generator.state == before


def test_covariance_rescaling_rejects_unrepresentable_values() -> None:
    with pytest.raises(ArithmeticError, match="below float64 range"):
        _sample_covariance(np.array([[0.0], [1e-200]]))
    with pytest.raises(ArithmeticError, match="exceeds float64 range"):
        _sample_covariance(np.array([[0.0], [1e200]]))


def test_pinned_icenreg_prescribed_bootstrap_coefficients_and_covariance() -> None:
    inputs = list(csv.DictReader((_FIXTURES / "interval-survival-inputs.csv").open(newline="")))
    resamples = _rows("resamples")
    coefficients = _rows("coefficients")
    covariance = _rows("covariance")
    metrics = _rows("metrics")
    for weight_case, input_case, prefix in (
        ("unit", "mixed", "unit"),
        ("weighted", "mixed_weighted", "weighted"),
    ):
        rows = [row for row in inputs if row["case"] == input_case]
        lower = np.array([float(row["lower"]) for row in rows])
        upper = np.array([float(row["upper"]) if row["upper"] != "Inf" else np.inf for row in rows])
        x = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
        weights = np.array([float(row["weight"]) for row in rows])
        names = (f"{prefix}_a", f"{prefix}_b", f"{prefix}_singular")
        tape = np.array(
            [
                [int(row["row_id_zero_based"]) for row in resamples if row["case"] == name]
                for name in names
            ],
            dtype=np.int64,
        )
        result = bootstrap_interval_survival_coefficients(
            lower,
            upper,
            x,
            weights=weights,
            replicates=len(names),
            resample_indices=tape,
        )
        assert result.resample_size == int(np.ceil(weights.sum()))
        assert result.replicate_status == ("ok", "ok", "fit_failed")
        for replicate, name in enumerate(names):
            metric = next(row for row in metrics if row["case"] == name)
            assert int(metric["bootstrap_draw_count"]) == result.resample_size
            assert int(metric["unique_rows"]) == np.unique(tape[replicate]).size
            expected = [row for row in coefficients if row["case"] == name]
            if replicate < 2:
                np.testing.assert_allclose(
                    result.coefficient_samples[replicate],
                    [float(row["value"]) for row in expected],
                    atol=2e-5,
                    rtol=2e-5,
                )
            else:
                assert all(row["status"] == "singular_design" for row in expected)
                assert np.isnan(result.coefficient_samples[replicate]).all()
        expected_covariance = np.array(
            [
                [
                    float(
                        next(
                            row["value"]
                            for row in covariance
                            if row["case"] == weight_case
                            and int(row["row"]) == i + 1
                            and int(row["column"]) == j + 1
                        )
                    )
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(result.covariance, expected_covariance, atol=3e-5, rtol=3e-5)
