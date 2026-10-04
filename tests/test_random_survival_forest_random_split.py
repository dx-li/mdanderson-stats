"""Source-ledger checks for the RF-SRC random split rule."""

from __future__ import annotations

import numpy as np

from mdanderson_stats.random_survival_forest import (
    _Budget,
    _grow_tree,
    _random_factor_group_probabilities,
    _random_factor_partition,
    fit_random_survival_forest,
)


class _IntegerTape:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = iter(values)

    def integers(self, high: int) -> int:
        value = next(self.values)
        assert 0 <= value < high
        return value


def _root_split(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    rng: object,
    *,
    mtry: int,
    categorical_columns: frozenset[int] = frozenset(),
):
    return _grow_tree(
        x,
        time,
        event,
        np.arange(time.size),
        rng=rng,  # type: ignore[arg-type]
        mtry=mtry,
        nodesize=2,
        nsplit=1,
        budget=_Budget(),
        max_nodes=100,
        max_split_work=10_000,
        max_leaf_records=100,
        categorical_columns=categorical_columns,
        split_rule="random",
        split_probability=None,
    )


def test_random_rule_uses_one_random_cut_not_the_best_survival_cut() -> None:
    x = np.arange(4.0)[:, None]
    tape = _IntegerTape((0, 2))  # only feature, cut after the third value
    event_a = np.array([1.0, 0.0, 1.0, 0.0])
    event_b = np.array([0.0, 1.0, 0.0, 1.0])
    time_a = np.array([1.0, 2.0, 3.0, 4.0])
    time_b = np.array([4.0, 1.0, 3.0, 2.0])

    first = _root_split(x, time_a, event_a, tape, mtry=1)
    second = _root_split(x, time_b, event_b, _IntegerTape((0, 2)), mtry=1)

    assert first.feature[0] == second.feature[0] == 0
    assert first.threshold[0] == second.threshold[0] == 2.0


def test_constant_feature_consumes_mtry_slot_before_next_feature() -> None:
    x = np.column_stack((np.ones(4), np.arange(4.0), np.arange(4.0)[::-1]))
    # Draw feature 0 (constant), then feature 1, then its threshold index 0.
    tree = _root_split(
        x,
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 0.0, 1.0, 0.0]),
        _IntegerTape((0, 0, 0)),
        mtry=2,
    )
    assert tree.feature[0] == 1
    assert tree.threshold[0] == 0.0


def test_full_mtry_orders_features_without_a_feature_draw() -> None:
    x = np.column_stack((np.arange(4.0), np.arange(4.0)[::-1]))
    tree = _root_split(
        x,
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 0.0, 1.0, 0.0]),
        _IntegerTape((1,)),  # only the cut draw; feature 0 is first by source order
        mtry=2,
    )
    assert tree.feature[0] == 0
    assert tree.threshold[0] == 1.0


def test_factor_partition_uses_complement_group_cardinalities() -> None:
    sizes4, probabilities4 = _random_factor_group_probabilities(4)
    sizes5, probabilities5 = _random_factor_group_probabilities(5)
    assert sizes4.tolist() == [1, 2]
    assert np.allclose(probabilities4, [4 / 7, 3 / 7], rtol=0, atol=1e-15)
    assert sizes5.tolist() == [1, 2]
    assert np.allclose(probabilities5, [1 / 3, 2 / 3], rtol=0, atol=1e-15)


def test_factor_partition_draw_order_is_group_then_uniform_subset() -> None:
    levels = np.arange(4, dtype=np.float64)
    sizes, probabilities = _random_factor_group_probabilities(levels.size)
    expected_rng = np.random.default_rng(7201)
    group_size = int(expected_rng.choice(sizes, p=probabilities))
    expected = np.sort(expected_rng.choice(levels, size=group_size, replace=False))

    actual = _random_factor_partition(levels, np.random.default_rng(7201))

    assert np.array_equal(actual, expected)
    assert actual.size in (1, 2)
    assert actual.size < levels.size


def test_public_random_rule_ignores_nsplit_candidate_count() -> None:
    x = np.arange(8.0)[:, None]
    time = np.arange(1.0, 9.0)
    event = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    first = fit_random_survival_forest(
        time,
        event,
        x,
        split_rule="random",
        n_trees=2,
        mtry=1,
        nodesize=1,
        nsplit=0,
        random_state=108,
    )
    second = fit_random_survival_forest(
        time,
        event,
        x,
        split_rule="random",
        n_trees=2,
        mtry=1,
        nodesize=1,
        nsplit=10,
        random_state=108,
    )
    assert first.split_rule == second.split_rule == "random"
    assert first.split_probability is second.split_probability is None
    for first_tree, second_tree in zip(first.trees, second.trees, strict=True):
        assert np.array_equal(first_tree.feature, second_tree.feature)
        assert np.array_equal(first_tree.threshold, second_tree.threshold, equal_nan=True)
