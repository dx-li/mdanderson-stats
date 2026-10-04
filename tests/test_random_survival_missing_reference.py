"""Independent fixed-tape references for source-defined missing-data rules."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.random_survival_forest import (
    _impute_node_values,
    _leaf_curve,
    _terminal_impute_value,
    fit_random_survival_forest,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class _TapeRng:
    """Tiny test-only RNG protocol adapter for explicit reference choices."""

    def __init__(
        self, *, integer_draws: list[int] | None = None, random_draws: list[float] | None = None
    ) -> None:
        self.integer_draws = list(integer_draws or [])
        self.random_draws = list(random_draws or [])

    def integers(self, high: int, size: int | None = None) -> int | np.ndarray:
        count = 1 if size is None else size
        draws = self.integer_draws[:count]
        del self.integer_draws[:count]
        assert len(draws) == count
        assert all(0 <= draw < high for draw in draws)
        if size is None:
            return draws[0]
        return np.asarray(draws, dtype=np.int64)

    def random(self) -> float:
        assert self.random_draws
        return self.random_draws.pop(0)


def test_node_donors_preserve_bootstrap_multiplicity_and_exclude_oob_rows() -> None:
    expected = _csv_rows("random-survival-missing-donors.csv")
    values = np.array([1.0, 9.0, 100.0, np.nan, np.nan])
    original_missing = np.isnan(values)
    donor_rows = np.array([0, 0, 1, 2], dtype=np.int64)
    recipient_rows = np.array([3, 4], dtype=np.int64)

    filled, selected = _impute_node_values(
        values,
        original_missing,
        donor_rows,
        recipient_rows,
        rng=_TapeRng(integer_draws=[0, 3]),  # source tape U=.125 and U=1
    )

    np.testing.assert_array_equal(filled, [float(row["imputed_value"]) for row in expected])
    np.testing.assert_array_equal(selected, np.array([0, 2]))
    assert not set(selected).intersection(recipient_rows)


def test_terminal_value_mean_snap_mode_ties_and_ancestor_fallback() -> None:
    snap = _csv_rows("random-survival-missing-time-snap.csv")
    values = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.zeros(values.size, dtype=bool)
    donor_rows = np.arange(values.size, dtype=np.int64)
    snapped = [
        _terminal_impute_value(
            values,
            observed,
            donor_rows,
            np.empty(0, dtype=np.int64),
            kind="time",
            master_times=np.array([2.0, 3.0]),
            rng=_TapeRng(random_draws=[float(u)]),
        )
        for row in snap
        for u in [float(row["tie_uniform"])]
    ]
    np.testing.assert_array_equal(snapped, [float(row["snapped_time"]) for row in snap])

    near_ties = _csv_rows("random-survival-missing-time-near-ties.csv")
    near_snapped = [
        _terminal_impute_value(
            np.array([float(row["value"])]),
            np.array([False]),
            np.array([0]),
            np.empty(0, dtype=np.int64),
            kind="time",
            master_times=np.array([1.0, 2.0, 3.0, 4.0]),
            rng=_TapeRng(random_draws=[float(row["uniform"])]),
        )
        for row in near_ties
    ]
    np.testing.assert_array_equal(near_snapped, [float(row["snapped_time"]) for row in near_ties])

    fallback_values = np.array([4.0, 6.0, 2.0, 8.0])
    missing = np.array([True, True, True, True])
    mode = _terminal_impute_value(
        fallback_values,
        missing,
        np.array([0, 1]),
        np.array([2, 3]),
        kind="categorical",
        master_times=np.empty(0),
        rng=_TapeRng(integer_draws=[1]),
    )
    assert mode == 8.0

    mode_rows = _csv_rows("random-survival-missing-terminal-mode.csv")
    mode_outputs = [
        _terminal_impute_value(
            fallback_values,
            missing,
            np.array([0, 1]),
            np.array([2, 3]),
            kind="categorical",
            master_times=np.empty(0),
            rng=_TapeRng(integer_draws=[index]),
        )
        for index, _row in enumerate(mode_rows)
    ]
    np.testing.assert_array_equal(mode_outputs, [float(row["selected_mode"]) for row in mode_rows])

    parent_values, parent_donors = _impute_node_values(
        fallback_values,
        missing,
        np.array([2, 3]),
        np.array([0, 1]),
        rng=_TapeRng(),
    )
    np.testing.assert_array_equal(parent_values, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(parent_donors, np.array([-1, -1]))


def test_candidate_specific_original_missingness_controls_observable_root_split() -> None:
    expected = _csv_rows("random-survival-missing-split-scores.csv")
    selected_cut = next(float(row["cut"]) for row in expected if row["selected"] == "TRUE")
    for seed in (0, 1, 2, 2718):
        fit = fit_random_survival_forest(
            time=np.array([1.0, 2.0, 3.0, 3.0, 5.0, 4.0]),
            event=np.array([1.0, 0.0, np.nan, 1.0, 0.0, 1.0]),
            covariates=np.array([[1.0], [2.0], [3.0], [4.0], [np.nan], [6.0]]),
            n_trees=1,
            nodesize=2,
            nsplit=0,
            replace=False,
            sample_fraction=1.0,
            ntime=None,
            random_state=seed,
            na_action="impute",
        )
        tree = fit.trees[0]
        assert tree.feature[0] == 0
        assert tree.threshold[0] == selected_cut
        # Output event grid comes from original complete events, not terminal
        # imputed event times; the latter may include master times absent here.
        np.testing.assert_array_equal(fit.time_grid, np.array([1.0, 3.0, 4.0]))


def test_terminal_km_counts_include_completed_time_and_status() -> None:
    expected = _csv_rows("random-survival-missing-terminal-counts.csv")
    time = np.array([1.0, 2.0, 3.0, 4.0, 2.0])
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    event_times, log_survival, cumulative_hazard = _leaf_curve(time, event)
    np.testing.assert_array_equal(event_times, [float(row["event_time"]) for row in expected])
    np.testing.assert_allclose(
        np.exp(log_survival),
        [float(row["survival_after_step"]) for row in expected],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        cumulative_hazard,
        np.cumsum([float(row["hazard_increment"]) for row in expected]),
        rtol=0,
        atol=1e-15,
    )
