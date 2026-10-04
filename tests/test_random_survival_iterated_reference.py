"""Compare iterative forest summaries with the independent fixed-tape ledger."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mdanderson_stats._random_survival_forest_imputation import pool_imputation_summaries

from mdanderson_stats.random_survival_forest import fit_random_survival_forest

FIXTURES = Path(__file__).parent / "fixtures"


class _UniformTape:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.position = 0

    def random(self) -> float:
        if self.position >= len(self.values):
            raise AssertionError("imputation consumed more random values than the source ledger")
        value = self.values[self.position]
        self.position += 1
        return value


def _csv_rows(path: str) -> list[dict[str, str]]:
    with (FIXTURES / path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _values(text: str) -> np.ndarray:
    return np.asarray([_number(value) for value in text.split(";")], dtype=np.float64)


def _number(text: str) -> float:
    return float("nan") if text == "NA" else float(text)


def test_coupled_oob_pool_matches_fixed_uniform_ledger() -> None:
    rows = _csv_rows("random-survival-iterated-coupled-rows.csv")
    tree_rows = _csv_rows("random-survival-iterated-coupled-trees.csv")
    tape_rows = _csv_rows("random-survival-iterated-coupled-tape.csv")
    grid_row = _csv_rows("random-survival-iterated-coupled-grids.csv")[0]

    missing_time = np.asarray([row["missing_time"] == "TRUE" for row in rows])
    missing_event = np.asarray([row["missing_event"] == "TRUE" for row in rows])
    original_time = np.asarray([_number(row["original_time"]) for row in rows])
    original_event = np.asarray([_number(row["original_event"]) for row in rows])
    # The helper receives a completed previous-pass array plus immutable masks.
    time = np.where(missing_time, 2.0, original_time)
    event = np.where(missing_event, 0.0, original_event)
    missing_covariates = np.zeros((len(rows), 1), dtype=bool)
    covariates = np.asarray([[float(row["original_row"])] for row in rows])
    leaves = np.asarray([int(row["leaf_node"]) for row in rows], dtype=np.int64)

    trees = []
    packed_membership = []
    for row in tree_rows:
        inbag = [int(value) for value in row["inbag_mask"].split(";")]
        packed_membership.append(sum(value << index for index, value in enumerate(inbag)))
        trees.append(
            SimpleNamespace(
                training_leaf_node=leaves,
                terminal_time=_values(row["terminal_time"]),
                terminal_event=_values(row["terminal_event"]),
                terminal_predictor=None,
            )
        )

    tape = _UniformTape([float(row["uniform"]) for row in tape_rows])
    result = pool_imputation_summaries(
        time,
        event,
        covariates,
        missing_time,
        missing_event,
        missing_covariates,
        trees,
        np.asarray(packed_membership, dtype=np.uint8).reshape(len(trees), 1),
        master_times=_values(grid_row["master_grid"]),
        rng=tape,  # type: ignore[arg-type]
        selection="oob",
    )

    np.testing.assert_array_equal(result.time, _values("1;3;3;9;3;4"))
    np.testing.assert_array_equal(result.event, _values("1;0;1;1;0;1"))
    np.testing.assert_array_equal(result.covariates, covariates)
    expected_fallback = np.zeros((len(rows), 3), dtype=bool)
    expected_fallback[2, :2] = True
    np.testing.assert_array_equal(result.fallback_used, expected_fallback)
    assert tape.position == len(tape_rows)
    assert not result.time.flags.writeable
    assert not result.event.flags.writeable
    assert not result.covariates.flags.writeable

    # An imputed event at time 9 changes the completed-data event set, but not
    # the original event-interest grid used by the fit.
    expected_grid = _values(grid_row["expected_fixed_event_grid"])
    rebuilt_grid = _values(grid_row["completed_event_grid_if_rebuilt"])
    assert 9.0 not in expected_grid
    assert 9.0 in rebuilt_grid


def test_iterated_fit_keeps_original_event_grid_and_row_mapping() -> None:
    time = np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan])
    event = np.asarray([1.0, 1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0, np.nan])
    covariates = np.asarray([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0], [7.0], [8.0], [np.nan]])
    fit = fit_random_survival_forest(
        time,
        event,
        covariates,
        na_action="impute",
        nimpute=2,
        n_trees=2,
        sample_fraction=0.875,
        replace=False,
        nodesize=1,
        nsplit=0,
        ntime=None,
        compute_oob=False,
        random_state=1,
    )

    np.testing.assert_array_equal(fit.training_row_indices, np.arange(8))
    np.testing.assert_array_equal(fit.time_grid, np.asarray([1.0, 2.0, 5.0, 7.0]))
    assert fit.requested_imputation_passes == 2
    assert fit.imputation_passes == 2
    assert fit.completed_time is not None and fit.completed_event is not None
    assert fit.completed_covariates is not None
    assert fit.completed_time.shape == (8,)
    assert fit.completed_event.shape == (8,)
    assert fit.completed_covariates.shape == (8, 1)
    assert fit.completed_event[3] == 1.0
    assert fit.completed_time[3] == 4.0
