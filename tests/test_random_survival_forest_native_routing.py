"""Small unchanged-C references for nominal masks and anti-split traversal."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats.random_survival_forest import (
    _factor_split_candidates,
    _factor_split_plan,
    _PackedTree,
    _tree_goes_left,
)
from mdanderson_stats.random_survival_forest_vimp import _route_anti_hazard


class _UniformTape:
    def __init__(self, value: float) -> None:
        self.value = value
        self.consumed = 0

    def random(self) -> float:
        self.consumed += 1
        return self.value


def test_native_subset_masks_and_anti_membership() -> None:
    fixture = Path(__file__).parent / "fixtures/random-survival-extensions-native.csv"
    with fixture.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for count in (3, 4):
        plan = _factor_split_plan(np.arange(count, dtype=float), 20, 0)
        assert plan[:2] == (2 ** (count - 1) - 1, True)
        candidates = list(
            _factor_split_candidates(np.arange(count), *plan, np.random.default_rng(1))
        )
        actual = [(len(levels), sum(1 << int(code) for code in levels)) for levels, _ in candidates]
        expected = [
            (int(row["b"]), int(row["c"]))
            for row in rows
            if row["kind"] == "enum" and int(row["a"]) == count
        ]
        assert actual == expected
    # Phase2 uses inclusive bounds with nsplit>0, but a strict row bound with nsplit=0.
    assert _factor_split_plan(np.arange(4), 7, 7)[:2] == (7, True)
    assert _factor_split_plan(np.arange(4), 7, 0)[:2] == (7, False)

    tree = _PackedTree(
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, np.nan, np.nan]),
        categorical_node=np.zeros(3, dtype=bool),
        split_level_offset=np.zeros(3, dtype=int),
        split_level_count=np.zeros(3, dtype=int),
        split_levels=np.array([], dtype=int),
        left=np.array([1, -1, -1]),
        right=np.array([2, -1, -1]),
        event_offset=np.array([0, 0, 1]),
        event_count=np.array([0, 1, 1]),
        event_time=np.array([1.0, 1.0]),
        log_survival=np.log(np.array([0.5, 0.25])),
        cumulative_hazard=np.array([1.0, 2.0]),
    )
    factor_tree = replace(
        tree,
        categorical_node=np.array([True, False, False]),
        split_level_count=np.array([3, 0, 0]),
        split_levels=np.array([0, 2, 3]),
    )
    for row in rows:
        if row["kind"] == "factor":
            assert _tree_goes_left(factor_tree, 0, float(row["a"]) - 1) == (int(row["b"]) == 1)
        elif row["kind"] == "anti":
            tape = _UniformTape(float(row["b"]))
            mortality = _route_anti_hazard(
                tree,
                np.array([0.25]),
                np.array([1.0]),
                0,
                threshold=float(row["a"]),
                rng=tape,  # type: ignore[arg-type]
            )
            assert mortality == float(row["c"])
            assert tape.consumed == int(row["d"])
