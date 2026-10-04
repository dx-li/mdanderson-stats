"""Compare OOB response completion and concordance with fixed R ledgers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.random_survival_forest import (
    _complete_oob_responses,
    _oob_concordance_error,
    _pack_tree,
)

FIXTURES = Path(__file__).parent / "fixtures"


class _IndexTape:
    def __init__(self, *indexes: int) -> None:
        self._indexes = iter(indexes)

    def integers(self, high: int, size: int | None = None) -> int | np.ndarray:
        count = 1 if size is None else size
        values = np.fromiter((next(self._indexes) for _ in range(count)), dtype=np.int64)
        if np.any((values < 0) | (values >= high)):
            raise AssertionError("tape index falls outside requested native selection pool")
        return int(values[0]) if size is None else values


def _read(name: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _tree(terminal_time: float, terminal_event: float, row_count: int):
    return _pack_tree(
        feature=[-1],
        threshold=[0.0],
        categorical_node=[False],
        split_level_offset=[0],
        split_level_count=[0],
        split_levels=[],
        left=[-1],
        right=[-1],
        event_offset=[0],
        event_count=[0],
        event_time=[],
        log_survival=[],
        cumulative_hazard=[],
        training_leaf_node=np.zeros(row_count, dtype=np.int32),
        terminal_time=[terminal_time],
        terminal_event=[terminal_event],
    )


def _membership(
    row_count: int,
    trees: int,
    recipient: int,
    *,
    oob: bool,
    inbag_outlier: bool = False,
) -> np.ndarray:
    matrix = np.ones((trees, row_count), dtype=np.uint8)
    matrix[:, recipient] = 0 if oob else 1
    if inbag_outlier:
        matrix[-1, recipient] = 1
    return np.packbits(matrix, axis=1, bitorder="little")


def _single_recipient_case(row: dict[str, str]) -> tuple[np.ndarray, ...]:
    n = int(row["recipient_id"])
    recipient = int(row["recipient_id"]) - 1
    time = np.linspace(1.0, float(n), n)
    event = np.asarray([float(i % 2) for i in range(n)])
    fallback_time_values = [
        float(value) for value in row["global_observed_fallback_values"].split(";") if value
    ]
    fallback_event_values = [
        float(value) for value in row["global_observed_fallback_values"].split(";") if value
    ]
    missing_time = np.zeros(n, dtype=bool)
    missing_event = np.zeros(n, dtype=bool)
    if row["field"] == "time":
        time[recipient] = np.nan
        missing_time[recipient] = True
    else:
        missing_event[recipient] = True
    values = [float(value) for value in row["oob_terminal_values"].split(";") if value]
    is_oob = bool(values)
    if row["field"] == "time":
        time[:4] = fallback_time_values
        terminal_times = values or [2.0, 8.0]
        if row["inbag_terminal_outlier"] not in ("", "NA"):
            terminal_times.append(float(row["inbag_terminal_outlier"]))
        terminal_events = [1.0] * len(terminal_times)
    else:
        event[:4] = fallback_event_values
        terminal_events = values or [0.0, 1.0]
        terminal_times = [1.0] * len(terminal_events)
    trees = tuple(_tree(t, e, n) for t, e in zip(terminal_times, terminal_events, strict=True))
    membership = _membership(
        n,
        len(trees),
        recipient,
        oob=is_oob,
        inbag_outlier=row["inbag_terminal_outlier"] not in ("", "NA"),
    )
    if not is_oob:
        if row["field"] == "time":
            time[:4] = [1.0, 3.0, 7.0, 9.0]
            time[recipient] = np.nan
        else:
            event[:4] = [0.0, 1.0, 0.0, 1.0]
    return time, event, missing_time, missing_event, trees, membership


def test_oob_missing_response_pools_match_fixed_tape_reference() -> None:
    expected_by_case = {
        row["case"]: row for row in _read("random-survival-missing-oob-responses.csv")
    }
    expected = {
        "mean_of_snapped_tree_times_left_unsnapped": 2,
        "status_mode_tie_low": 0,
        "status_mode_tie_high": 1,
        "empty_oob_time_global_fallback": 3,
        "empty_oob_status_global_fallback": 1,
    }
    for case, target in expected.items():
        row = expected_by_case[case]
        time, event, missing_time, missing_event, trees, membership = _single_recipient_case(row)
        tape = []
        if row["uniform"] not in ("", "NA"):
            if row["pool"] == "oob_tree_terminal":
                # Native mode selection samples among sorted tied modes.
                high = len(set(row["oob_terminal_values"].split(";")))
            else:
                high = len(row["global_observed_fallback_values"].split(";"))
            tape.append(int(np.ceil(float(row["uniform"]) * high)) - 1)
        completed_time, completed_event, fallback = _complete_oob_responses(
            time,
            event,
            missing_time,
            missing_event,
            trees,
            membership,
            rng=_IndexTape(*tape),  # type: ignore[arg-type]
            max_cells=1_000,
            max_work=1_000,
        )
        recipient = int(row["recipient_id"]) - 1
        value = completed_time[recipient] if row["field"] == "time" else completed_event[recipient]
        assert value == target == float(row["expected"])
        if case == "mean_of_snapped_tree_times_left_unsnapped":
            observed_times = [
                float(value) for value in row["global_observed_fallback_values"].split(";") if value
            ]
            assert value not in observed_times
        assert bool(fallback[recipient, 0 if row["field"] == "time" else 1]) == (
            row["fallback_used"] == "TRUE"
        )


def test_oob_response_completion_preserves_observed_component() -> None:
    row = _read("random-survival-missing-oob-preservation.csv")[0]
    n = 10
    recipient = int(row["row_id"]) - 1
    time = np.arange(1.0, n + 1.0)
    time[recipient] = np.nan
    event = np.asarray([float(i % 2) for i in range(n)])
    event[recipient] = float(row["event_original"])
    missing_time = np.zeros(n, dtype=bool)
    missing_event = np.zeros(n, dtype=bool)
    missing_time[recipient] = True
    trees = (_tree(1.0, 0.0, n), _tree(3.0, 1.0, n))
    completed_time, completed_event, fallback = _complete_oob_responses(
        time,
        event,
        missing_time,
        missing_event,
        trees,
        _membership(n, 2, recipient, oob=True),
        rng=_IndexTape(),  # type: ignore[arg-type]
        max_cells=1_000,
        max_work=1_000,
    )
    assert completed_time[recipient] == float(row["expected_time"])
    assert completed_event[recipient] == float(row["expected_event"])
    assert not fallback[recipient].any()


def test_status_pool_ties_precede_time_global_fallback_draws() -> None:
    time = np.asarray([1.0, 3.0, np.nan, 7.0])
    event = np.asarray([0.0, 1.0, 0.0, np.nan])
    missing_time = np.asarray([False, False, True, False])
    missing_event = np.asarray([False, False, False, True])
    trees = (
        _pack_tree(
            feature=[-1, -1],
            threshold=[0.0, 0.0],
            categorical_node=[False, False],
            split_level_offset=[0, 0],
            split_level_count=[0, 0],
            split_levels=[],
            left=[-1, -1],
            right=[-1, -1],
            event_offset=[0, 0],
            event_count=[0, 0],
            event_time=[],
            log_survival=[],
            cumulative_hazard=[],
            training_leaf_node=np.asarray([0, 0, 0, 1], dtype=np.int32),
            terminal_time=[1.0, 3.0],
            terminal_event=[0.0, 0.0],
        ),
        _pack_tree(
            feature=[-1, -1],
            threshold=[0.0, 0.0],
            categorical_node=[False, False],
            split_level_offset=[0, 0],
            split_level_count=[0, 0],
            split_levels=[],
            left=[-1, -1],
            right=[-1, -1],
            event_offset=[0, 0],
            event_count=[0, 0],
            event_time=[],
            log_survival=[],
            cumulative_hazard=[],
            training_leaf_node=np.asarray([0, 0, 0, 1], dtype=np.int32),
            terminal_time=[1.0, 3.0],
            terminal_event=[0.0, 1.0],
        ),
    )
    membership = np.packbits(
        np.asarray([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=np.uint8),
        axis=1,
        bitorder="little",
    )
    draw_rows = _read("random-survival-missing-oob-rng-order.csv")
    tape = [
        int(np.ceil(float(row["uniform"]) * len(row["pool"].split(";")))) - 1 for row in draw_rows
    ]
    assert tape == [1, 0]
    completed_time, completed_event, fallback = _complete_oob_responses(
        time,
        event,
        missing_time,
        missing_event,
        trees,
        membership,
        rng=_IndexTape(*tape),  # status tie first, then time fallback
        max_cells=1_000,
        max_work=1_000,
    )  # type: ignore[arg-type]
    assert [row["operation"] for row in draw_rows] == [
        "status_tied_mode",
        "time_global_fallback",
    ]
    assert completed_event[3] == float(draw_rows[0]["expected_value"])
    assert completed_time[2] == float(draw_rows[1]["expected_value"])
    assert fallback[2, 0]
    assert not fallback[3, 1]


def test_oob_native_concordance_matches_pairwise_reference() -> None:
    rows = _read("random-survival-missing-oob-concordance-input.csv")
    time = np.asarray([float(row["time"]) for row in rows])
    event = np.asarray([float(row["event"]) for row in rows])
    risk = np.asarray([float(row["risk"]) for row in rows])
    contributors = np.asarray([int(row["contributors"]) for row in rows])
    expected = _read("random-survival-missing-oob-concordance-summary.csv")[0]
    error, comparable = _oob_concordance_error(time, event, risk, contributors)
    assert comparable == int(expected["comparable_unordered_pairs"])
    assert np.isclose(error, float(expected["concordance_error"]), rtol=0, atol=1e-15)


def test_no_contributing_oob_rows_yields_nan_and_zero_pairs() -> None:
    error, comparable = _oob_concordance_error(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 2.0]),
        np.zeros(2, dtype=np.int64),
    )
    assert np.isnan(error)
    assert comparable == 0
