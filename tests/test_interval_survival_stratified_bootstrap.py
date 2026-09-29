from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.interval_survival_stratified_bootstrap import (
    bootstrap_stratified_interval_survival_coefficients,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    upper = np.array([1, np.inf, 1, np.inf, 1, np.inf, 1, np.inf])
    x = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)[:, None]
    strata = np.repeat(np.array(["A", "B"]), 4)
    weights = np.array([2, 8, 5, 5, 4, 6, 7, 3], dtype=float)
    return lower, upper, x, strata, weights


def _read_fixture_tape() -> np.ndarray:
    with (_FIXTURES / "interval-stratified-bootstrap-resamples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    tape = np.empty((6, 40), dtype=np.int64)
    for row in rows:
        tape[int(row["replicate"]), int(row["position"])] = int(row["source_index"])
    return tape


def test_fixed_within_stratum_tapes_match_independent_cloglog_reference() -> None:
    lower, upper, x, strata, weights = _fixture()
    tape = _read_fixture_tape()
    result = bootstrap_stratified_interval_survival_coefficients(
        lower,
        upper,
        x,
        strata=strata,
        weights=weights,
        replicates=6,
        resampling="within_stratum",
        resample_indices=tape,
    )

    assert result.stratum_labels == ("A", "B")
    assert result.within_stratum_sample_sizes == (20, 20)
    assert result.resample_size == 40
    np.testing.assert_array_equal(result.resample_indices, tape)
    np.testing.assert_array_equal(result.resampled_stratum_counts, [[20, 20]] * 6)
    assert result.replicate_status == ("ok",) * 5 + ("fit_failed",)
    assert result.successful_replicates == 5
    assert result.failed_replicates == 1
    assert result.covariance_conditional_on_success
    assert not result.resample_indices.flags.writeable

    with (_FIXTURES / "interval-stratified-bootstrap-coefficients.csv").open(newline="") as stream:
        coefficients = list(csv.DictReader(stream))
    expected = np.array([float(row["beta"]) for row in coefficients])
    np.testing.assert_allclose(result.coefficient_samples[:5, 0], expected[:5], atol=4e-6, rtol=0)
    assert np.isnan(result.coefficient_samples[5, 0])
    with (_FIXTURES / "interval-stratified-bootstrap-summary.csv").open(newline="") as stream:
        summary = {row["quantity"]: float(row["value"]) for row in csv.DictReader(stream)}
    assert result.fit.coefficients[0] == pytest.approx(summary["original_beta"], abs=3e-7)
    assert result.fit.log_likelihood == pytest.approx(summary["original_log_likelihood"], abs=3e-7)
    assert result.covariance[0, 0] == pytest.approx(summary["covariance"], abs=3e-6)
    assert result.standard_error[0] == pytest.approx(summary["standard_error"], abs=3e-6)


def test_weighted_within_stratum_sample_sizes_replay_and_failure_preflight() -> None:
    lower, upper, x, strata, weights = _fixture()
    first = bootstrap_stratified_interval_survival_coefficients(
        lower,
        upper,
        x,
        strata=strata,
        weights=weights * 0.26,
        replicates=2,
        resampling="within_stratum",
        rng=812,
    )
    replay = bootstrap_stratified_interval_survival_coefficients(
        lower,
        upper,
        x,
        strata=strata,
        weights=weights * 0.26,
        replicates=2,
        resampling="within_stratum",
        resample_indices=first.resample_indices,
    )
    assert first.within_stratum_sample_sizes == (6, 6)
    assert first.resample_size == 12
    np.testing.assert_array_equal(first.resample_indices, replay.resample_indices)
    np.testing.assert_allclose(
        first.coefficient_samples, replay.coefficient_samples, equal_nan=True
    )
    with pytest.raises(ValueError, match="within_stratum.*matching stratum"):
        bootstrap_stratified_interval_survival_coefficients(
            lower,
            upper,
            x,
            strata=strata,
            weights=weights,
            replicates=1,
            resampling="within_stratum",
            resample_indices=np.array([[4] * 20 + [0] * 20]),
        )


def test_pooled_scheme_uses_global_weighted_row_count_and_records_group_counts() -> None:
    lower, upper, x, strata, weights = _fixture()
    result = bootstrap_stratified_interval_survival_coefficients(
        lower,
        upper,
        x,
        strata=strata,
        weights=weights * 0.26,
        replicates=3,
        resampling="pooled",
        rng=101,
    )
    assert result.resample_size == 11
    assert result.resample_indices.shape == (3, 11)
    assert result.resampled_stratum_counts.shape == (3, 2)
    np.testing.assert_array_equal(result.resampled_stratum_counts.sum(axis=1), [11] * 3)
    assert result.within_stratum_sample_sizes == ()
    assert np.all((0 <= result.resample_indices) & (result.resample_indices < lower.size))

    omitted = bootstrap_stratified_interval_survival_coefficients(
        lower,
        upper,
        x,
        strata=strata,
        weights=weights * 0.26,
        replicates=1,
        resampling="pooled",
        resample_indices=np.zeros((1, 11), dtype=np.int64),
    )
    assert omitted.replicate_status == ("fit_failed",)
    np.testing.assert_array_equal(omitted.resampled_stratum_counts, [[11, 0]])
    assert "omitted strata ('B',)" in (omitted.replicate_errors[0] or "")


def test_explicit_policy_and_complex_tape_values_are_rejected() -> None:
    lower, upper, x, strata, weights = _fixture()
    with pytest.raises(ValueError, match="resampling must"):
        bootstrap_stratified_interval_survival_coefficients(
            lower,
            upper,
            x,
            strata=strata,
            weights=weights,
            resampling="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="integers"):
        bootstrap_stratified_interval_survival_coefficients(
            lower,
            upper,
            x,
            strata=strata,
            weights=weights,
            resampling="within_stratum",
            resample_indices=np.ones((1, 40), dtype=np.complex128),
            replicates=1,
        )
