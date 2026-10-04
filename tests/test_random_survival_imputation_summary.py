from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mdanderson_stats._random_survival_forest_imputation import pool_imputation_summaries


def _tree(
    leaves: list[int],
    *,
    time: list[float],
    event: list[float],
    predictors: list[list[float]],
) -> SimpleNamespace:
    return SimpleNamespace(
        training_leaf_node=np.asarray(leaves, dtype=np.int32),
        terminal_time=np.asarray(time, dtype=np.float64),
        terminal_event=np.asarray(event, dtype=np.float64),
        terminal_predictor=np.asarray(predictors, dtype=np.float64),
    )


def _fixture() -> tuple[object, ...]:
    time = np.asarray([0.0, 5.0, 10.0])
    event = np.asarray([1.0, 0.0, 1.0])
    x = np.asarray([[0.0, 0.0], [8.0, 0.0], [1.0, 1.0]])
    missing_time = np.asarray([False, True, False])
    missing_event = np.asarray([False, True, False])
    missing_x = np.asarray([[False, False], [True, True], [False, False]])
    trees = (
        _tree([0, 1, 0], time=[0.0, 4.0], event=[1.0, 0.0], predictors=[[0.0, 0.0], [2.0, 0.0]]),
        _tree([0, 1, 0], time=[0.0, 8.0], event=[1.0, 1.0], predictors=[[0.0, 0.0], [6.0, 1.0]]),
    )
    # Row 1 is out of bag in both trees; rows 0 and 2 are in bag.
    membership = np.asarray([[0b00000101], [0b00000101]], dtype=np.uint8)
    return time, event, x, missing_time, missing_event, missing_x, trees, membership


def test_pooling_uses_oob_tree_summaries_and_preserves_observed_values() -> None:
    args = _fixture()
    result = pool_imputation_summaries(
        *args[:6],
        args[6],
        args[7],
        master_times=np.asarray([0.0, 9.0, 10.0]),
        rng=np.random.default_rng(41),
        categorical_features=(1,),
        selection="oob",
    )

    assert result.time.tolist() == [0.0, 9.0, 10.0]
    assert result.event[0] == 1.0 and result.event[2] == 1.0
    assert result.event[1] in (0.0, 1.0)
    assert result.covariates[:, 0].tolist() == [0.0, 4.0, 1.0]
    assert result.covariates[1, 1] in (0.0, 1.0)
    assert not result.fallback_used.any()
    assert result.work_units == 48
    assert not result.time.flags.writeable
    assert not result.event.flags.writeable
    assert not result.covariates.flags.writeable


def test_empty_local_pool_uses_original_donors_and_marks_fallback() -> None:
    args = _fixture()
    membership = np.asarray([[0b00000111], [0b00000111]], dtype=np.uint8)
    result = pool_imputation_summaries(
        *args[:6],
        args[6],
        membership,
        master_times=np.asarray([0.0, 5.0, 10.0]),
        rng=np.random.default_rng(9),
        categorical_features=(1,),
        selection="oob",
    )

    assert result.time[1] in (0.0, 10.0)
    assert result.event[1] in (1.0,)
    assert result.covariates[1, 0] in (0.0, 1.0)
    assert result.covariates[1, 1] in (0.0, 1.0)
    assert result.fallback_used[1].all()
    assert not result.fallback_used[[0, 2]].any()


def test_all_tree_selection_and_budget_preflight() -> None:
    args = _fixture()
    all_result = pool_imputation_summaries(
        *args[:6],
        args[6],
        args[7],
        master_times=np.asarray([0.0, 9.0, 10.0]),
        rng=np.random.default_rng(2),
        categorical_features=(1,),
        selection="all",
    )
    assert all_result.time[1] == 9.0
    assert all_result.event[1] in (0.0, 1.0)
    assert all_result.covariates[1, 0] == pytest.approx(4.0)

    rng = np.random.default_rng(2)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_work"):
        pool_imputation_summaries(
            *args[:6],
            args[6],
            args[7],
            master_times=np.asarray([0.0, 9.0, 10.0]),
            rng=rng,
            categorical_features=(1,),
            selection="all",
            max_work=1,
        )
    assert rng.bit_generator.state == before


def test_master_time_snap_uses_asymmetric_near_midpoint_tie_rule() -> None:
    class Tape:
        def __init__(self) -> None:
            self.calls = 0

        def random(self) -> float:
            self.calls += 1
            return 0.0

    def complete(value: float, tape: Tape) -> float:
        tree = _tree([0, 1], time=[0.0, value], event=[1.0, 1.0], predictors=[[0.0], [0.0]])
        result = pool_imputation_summaries(
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([[0.0], [0.0]]),
            np.asarray([False, True]),
            np.asarray([False, False]),
            np.asarray([[False], [False]]),
            (tree,),
            np.asarray([[0b00000001]], dtype=np.uint8),
            master_times=np.asarray([0.0, 1.0]),
            rng=tape,
        )
        return float(result.time[1])

    below = Tape()
    assert complete(0.5 - 5e-11, below) == 0.0
    assert below.calls == 0
    above = Tape()
    assert complete(0.5 + 5e-11, above) == 0.0
    assert above.calls == 1
