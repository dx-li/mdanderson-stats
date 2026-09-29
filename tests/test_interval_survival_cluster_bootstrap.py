from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.interval_survival import fit_interval_survival
from mdanderson_stats.interval_survival_cluster_bootstrap import (
    bootstrap_interval_survival_cluster_coefficients,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with (_FIXTURES / "interval-survival-inputs.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case"] == "mixed"]
    lower = np.array([float(row["lower"]) for row in rows])
    upper = np.array([float(row["upper"]) if row["upper"] != "Inf" else np.inf for row in rows])
    x = np.array([[float(row["x1"]), float(row["x2"])] for row in rows])
    return lower, upper, x, np.arange(lower.size)


def test_frequency_collapsing_matches_expanded_unequal_cluster_rows() -> None:
    lower, upper, x, _ = _data()
    sizes = (18, 22, 26, 30)
    cluster_ids = np.repeat(np.array([20, 10, 40, 30]), sizes)
    tape = np.array([[0, 1, 1, 3], [2, 0, 3, 0]], dtype=np.int64)
    result = bootstrap_interval_survival_cluster_coefficients(
        lower,
        upper,
        cluster_ids,
        x,
        replicates=2,
        cluster_draw_indices=tape,
    )

    assert result.cluster_labels == (10, 20, 30, 40)
    np.testing.assert_array_equal(result.cluster_draw_indices, tape)
    np.testing.assert_array_equal(result.resampled_row_counts, [84, 100])
    assert result.replicate_status == ("ok", "ok")
    assert result.successful_replicates == 2
    assert result.covariance_conditional_on_success
    assert not result.cluster_draw_indices.flags.writeable

    # Independent unchanged icenReg optimizer on the same expanded row tapes.
    with (_FIXTURES / "interval-survival-cluster-coefficients.csv").open() as stream:
        native = np.array([float(row["value"]) for row in csv.DictReader(stream)]).reshape(2, 2)
    with (_FIXTURES / "interval-survival-cluster-covariance.csv").open() as stream:
        native_covariance = np.array(
            [float(row["value"]) for row in csv.DictReader(stream)]
        ).reshape(2, 2, order="F")
    np.testing.assert_allclose(result.coefficient_samples, native, rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(result.covariance, native_covariance, rtol=2e-5, atol=5e-8)

    group_rows = [
        np.arange(start, stop)
        for start, stop in zip(np.cumsum((0,) + sizes[:-1]), np.cumsum(sizes))
    ]
    sorted_group_rows = (group_rows[1], group_rows[0], group_rows[3], group_rows[2])
    for replicate, draw in enumerate(tape):
        expanded = np.concatenate([sorted_group_rows[int(group)] for group in draw])
        expanded_fit = fit_interval_survival(lower[expanded], upper[expanded], x[expanded])
        np.testing.assert_allclose(
            result.coefficient_samples[replicate], expanded_fit.coefficients, rtol=2e-6, atol=2e-7
        )


def test_cluster_relabeling_preserves_sorted_tape_mapping() -> None:
    lower, upper, x, _ = _data()
    original = np.repeat(np.array([20, 10, 40, 30]), (18, 22, 26, 30))
    relabeled = np.repeat(np.array(["b", "a", "d", "c"]), (18, 22, 26, 30))
    tape = np.array([[0, 1, 1, 3]], dtype=np.int64)
    numeric = bootstrap_interval_survival_cluster_coefficients(
        lower, upper, original, x, replicates=1, cluster_draw_indices=tape
    )
    textual = bootstrap_interval_survival_cluster_coefficients(
        lower, upper, relabeled, x, replicates=1, cluster_draw_indices=tape
    )
    np.testing.assert_array_equal(numeric.resampled_row_counts, textual.resampled_row_counts)
    np.testing.assert_allclose(numeric.coefficient_samples, textual.coefficient_samples)


def test_singular_single_cluster_draw_raises_or_records_without_redraw() -> None:
    lower, upper, x, _ = _data()
    cluster_ids = np.arange(lower.size)
    tape = np.zeros((1, lower.size), dtype=np.int64)
    with pytest.raises(ValueError, match="replicate 0 failed"):
        bootstrap_interval_survival_cluster_coefficients(
            lower,
            upper,
            cluster_ids,
            x,
            replicates=1,
            cluster_draw_indices=tape,
        )

    result = bootstrap_interval_survival_cluster_coefficients(
        lower,
        upper,
        cluster_ids,
        x,
        replicates=1,
        cluster_draw_indices=tape,
        on_fit_failure="record",
    )
    assert result.replicate_status == ("invalid_data",)
    assert result.failed_replicates == 1
    assert np.isnan(result.coefficient_samples).all()
    assert np.isnan(result.covariance).all()


def test_work_preflight_does_not_consume_rng_state() -> None:
    lower, upper, x, _ = _data()
    cluster_ids = np.repeat(np.arange(4), 24)
    rng = np.random.default_rng(194)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="worst-case cluster bootstrap fit work"):
        bootstrap_interval_survival_cluster_coefficients(
            lower,
            upper,
            cluster_ids,
            x,
            replicates=5_000,
            rng=rng,
            max_iterations=5_000,
        )
    assert rng.bit_generator.state == before
