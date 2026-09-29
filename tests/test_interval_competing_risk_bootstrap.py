from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.interval_competing_risk import _choose_knots, fit_interval_competing_risk
from mdanderson_stats.interval_competing_risk_bootstrap import (
    bootstrap_interval_competing_risk_coefficients,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv("interval-competing-risk-inputs.csv")
    lower = np.array([float(row["v"]) for row in rows])
    upper = np.array([float(row["u"]) if row["u"] != "Inf" else np.inf for row in rows])
    event = np.array([int(row["event"]) for row in rows])
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
    return lower, upper, event, x


def test_fixed_tape_preserves_duplicates_and_recomputes_knots() -> None:
    lower, upper, event, x = _inputs()
    tape_rows = _read_csv("interval-competing-risk-bootstrap-tapes.csv")
    tape = np.array(
        [
            [int(row["row_id"]) - 1 for row in tape_rows if int(row["case"]) == case]
            for case in (1, 2)
        ],
        dtype=np.int64,
    )
    expected_support = _read_csv("interval-competing-risk-bootstrap-support.csv")
    result = bootstrap_interval_competing_risk_coefficients(
        lower,
        upper,
        event,
        x,
        alpha=(0.0, 1.0),
        k=0.5,
        replicates=2,
        resample_indices=tape,
    )

    assert result.resample_size == lower.size
    assert result.replicate_status == ("ok", "ok")
    assert result.successful_replicates == 2
    assert result.failed_replicates == 0
    assert result.coefficient_samples.shape == (2, 4)
    assert np.isnan(result.fit.covariance).all()
    assert result.fit.converged
    assert not result.coefficient_samples.flags.writeable
    assert not result.covariance.flags.writeable
    np.testing.assert_allclose(
        result.covariance,
        np.cov(result.coefficient_samples, rowvar=False, ddof=1),
        rtol=2e-14,
        atol=2e-14,
    )
    single = bootstrap_interval_competing_risk_coefficients(
        lower,
        upper,
        event,
        x,
        alpha=(0.0, 1.0),
        k=0.5,
        replicates=1,
        resample_indices=tape[:1],
    )
    assert single.successful_replicates == 1
    assert np.isnan(single.covariance).all()
    assert np.isnan(single.standard_error).all()
    for index, case in enumerate((1, 2), start=1):
        sampled = tape[index - 1]
        python_knots, python_bounds = _choose_knots(
            np.r_[lower[sampled], upper[sampled][event[sampled] > 0]], 0.5
        )
        refit = fit_interval_competing_risk(
            lower[sampled],
            upper[sampled],
            event[sampled],
            x[sampled],
            alpha=(0.0, 1.0),
            k=0.5,
        )
        np.testing.assert_allclose(result.coefficient_samples[index - 1], refit.coefficients)
        assert refit.converged and refit.kkt_error < 2e-5
        np.testing.assert_array_equal(refit.knots, python_knots)
        np.testing.assert_array_equal(refit.boundary_knots, python_bounds)
        ref_rows = [row for row in expected_support if int(row["case"]) == case]
        reference_knots = np.array([float(row["knot"]) for row in ref_rows])
        np.testing.assert_allclose(refit.knots, reference_knots, atol=1e-8, rtol=0.0)


def test_missing_cause_aborts_by_default_or_is_explicitly_recorded() -> None:
    lower, upper, event, x = _inputs()
    cause_one = int(np.flatnonzero(event == 1)[0])
    tape = np.full((1, lower.size), cause_one, dtype=np.int64)
    with pytest.raises(ValueError, match="resample 0 omits cause 2"):
        bootstrap_interval_competing_risk_coefficients(
            lower,
            upper,
            event,
            x,
            alpha=(0.0, 1.0),
            k=0.5,
            replicates=1,
            resample_indices=tape,
        )

    result = bootstrap_interval_competing_risk_coefficients(
        lower,
        upper,
        event,
        x,
        alpha=(0.0, 1.0),
        k=0.5,
        replicates=1,
        resample_indices=tape,
        on_invalid_resample="record",
    )
    assert result.on_invalid_resample == "record"
    assert result.replicate_status == ("missing_cause",)
    assert result.failed_replicates == 1
    assert np.isnan(result.coefficient_samples).all()
    assert np.isnan(result.covariance).all()
    assert np.isnan(result.standard_error).all()


def test_fit_work_preflight_does_not_consume_generator_state() -> None:
    lower, upper, event, x = _inputs()
    rng = np.random.default_rng(902)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="worst-case bootstrap fitting work"):
        bootstrap_interval_competing_risk_coefficients(
            lower,
            upper,
            event,
            x,
            alpha=(0.0, 1.0),
            k=0.5,
            replicates=5_000,
            max_iterations=2_000,
            rng=rng,
        )
    assert rng.bit_generator.state == before


def test_bootstrap_requires_covariate_and_records_invalid_policy() -> None:
    lower, upper, event, _ = _inputs()
    with pytest.raises(ValueError, match="at least one covariate"):
        bootstrap_interval_competing_risk_coefficients(
            lower, upper, event, alpha=(0.0, 1.0), k=0.5, replicates=1
        )
    with pytest.raises(ValueError, match="on_invalid_resample"):
        bootstrap_interval_competing_risk_coefficients(
            lower,
            upper,
            event,
            np.arange(lower.size),
            alpha=(0.0, 1.0),
            k=0.5,
            replicates=1,
            on_invalid_resample="drop",
        )
