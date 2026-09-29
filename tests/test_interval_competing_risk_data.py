"""Longitudinal visit conversion for interval competing risks."""

import numpy as np
import pytest

from mdanderson_stats.interval_competing_risk_data import (
    prepare_interval_competing_risk_visits,
)


def test_visit_conversion_uses_chronology_and_preserves_row_provenance() -> None:
    result = prepare_interval_competing_risk_visits(
        subject_id=["b", "a", "b", None, "a", "b", "c"],
        visit_time=[2.0, 3.0, 1.0, np.nan, 4.0, 5.0, 7.0],
        status=[1, 2, 0, 0, 1, 9, 0],
        covariates=[[20.0], [30.0], [10.0], [np.nan], [40.0], [50.0], [70.0]],
    )

    assert result.subject_id == ("a", "b", "c")
    np.testing.assert_array_equal(result.lower, [0.0, 1.0, 7.0])
    np.testing.assert_array_equal(result.upper, [3.0, 2.0, np.inf])
    np.testing.assert_array_equal(result.event, [2, 1, 0])
    np.testing.assert_array_equal(result.covariates[:, 0], [30.0, 10.0, 70.0])
    np.testing.assert_array_equal(result.dropped_missing_time_rows, [3])
    np.testing.assert_array_equal(result.ignored_post_event_rows, [4, 5])
    np.testing.assert_array_equal(result.lower_source_row, [-1, 2, 6])
    np.testing.assert_array_equal(result.upper_source_row, [1, 0, -1])
    np.testing.assert_array_equal(result.covariate_source_row, [1, 2, 6])
    assert not result.lower.flags.writeable
    assert not result.covariates.flags.writeable


def test_visit_validation_rejects_ambiguous_ties_and_zero_length_events() -> None:
    with pytest.raises(ValueError, match="tied visit times"):
        prepare_interval_competing_risk_visits([1, 1], [1.0, 1.0], [0, 1])

    with pytest.raises(ValueError, match="positive-length interval"):
        prepare_interval_competing_risk_visits([1], [0.0], [1])

    with pytest.raises(ValueError, match="status must be 0, 1, or 2"):
        prepare_interval_competing_risk_visits([1], [1.0], [3])


def test_visit_conversion_drops_missing_time_before_identifier_validation() -> None:
    result = prepare_interval_competing_risk_visits([None, "kept"], [np.nan, 2.0], [np.nan, 0])
    assert result.subject_id == ("kept",)
    np.testing.assert_array_equal(result.dropped_missing_time_rows, [0])
    np.testing.assert_array_equal(result.lower, [2.0])
    assert np.isinf(result.upper[0])


def test_numeric_identifier_identity_is_not_lost_and_mixed_types_reject() -> None:
    large_integer = 2**53 + 1
    adjacent_float = float(2**53)
    result = prepare_interval_competing_risk_visits(
        [large_integer, adjacent_float], [1.0, 1.0], [0, 0]
    )
    assert result.subject_id == (adjacent_float, large_integer)
    np.testing.assert_array_equal(result.lower, [1.0, 1.0])

    with pytest.raises(ValueError, match="one sortable type"):
        prepare_interval_competing_risk_visits([1, "1"], [1.0, 1.0], [0, 0])


def test_portable_reference_fixture_matches_corrected_visit_ledger() -> None:
    import csv
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "fixtures"
    with (fixture_dir / "intccr-visits-input.csv").open(newline="") as stream:
        source = list(csv.DictReader(stream))
    result = prepare_interval_competing_risk_visits(
        [int(row["id"]) for row in source],
        [np.nan if row["time"] == "NA" else float(row["time"]) for row in source],
        [int(row["event"]) for row in source],
        [[float(row["x"])] for row in source],
    )
    with (fixture_dir / "intccr-visits-expected.csv").open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    assert result.subject_id == tuple(int(row["subject_id"]) for row in expected)
    for field, actual in (
        ("lower", result.lower),
        ("upper", result.upper),
        ("event", result.event),
        ("covariate", result.covariates[:, 0]),
        ("lower_source_row", result.lower_source_row),
        ("upper_source_row", result.upper_source_row),
        ("covariate_source_row", result.covariate_source_row),
    ):
        expected_values = [np.inf if row[field] == "Inf" else float(row[field]) for row in expected]
        np.testing.assert_array_equal(actual, expected_values)

    with (fixture_dir / "intccr-visits-exclusions.csv").open(newline="") as stream:
        exclusions = list(csv.DictReader(stream))
    dropped = [int(row["row_index"]) for row in exclusions if row["kind"] == "missing_visit_time"]
    ignored = [int(row["row_index"]) for row in exclusions if row["kind"] == "ignored_post_event"]
    np.testing.assert_array_equal(result.dropped_missing_time_rows, dropped)
    np.testing.assert_array_equal(result.ignored_post_event_rows, ignored)

    with (fixture_dir / "intccr-visits-native-sorted.csv").open(newline="") as stream:
        native_sorted = list(csv.DictReader(stream))
    assert [int(row["id"]) for row in native_sorted] == [1, 2, 3, 4]
    assert result.subject_id == (1, 2, 3, 4, 6)


def test_nested_visit_arrays_reject_before_numeric_conversion() -> None:
    with pytest.raises(ValueError, match="at most two-dimensional"):
        prepare_interval_competing_risk_visits([1, 1], [1.0, 2.0], [0, 0], [[[1.0]], [[2.0]]])

    with pytest.raises(ValueError, match="at most two-dimensional"):
        prepare_interval_competing_risk_visits([1], [1.0], [0], [np.array([[1.0, 2.0]])])
