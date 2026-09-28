"""Bounded OOB curves, membership and native concordance conventions."""

from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import (
    _oob_concordance_error,
    fit_random_survival_forest,
    predict_random_survival_forest,
)


def test_oob_membership_curves_and_default_predictions() -> None:
    time = np.arange(1.0, 13.0)
    event = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    x = np.column_stack((np.arange(12.0), np.arange(12.0) % 3))
    ordinary = fit_random_survival_forest(
        time, event, x, n_trees=9, sample_fraction=0.5, random_state=811
    )
    with_oob = fit_random_survival_forest(
        time, event, x, n_trees=9, sample_fraction=0.5, random_state=811, compute_oob=True
    )
    assert ordinary.inbag_membership is None
    assert ordinary.oob is None
    assert with_oob.inbag_membership is not None
    assert with_oob.oob is not None
    assert not with_oob.inbag_membership.flags.writeable
    assert not with_oob.oob.contributor_count.flags.writeable
    inbag = np.unpackbits(with_oob.inbag_membership, axis=1, bitorder="little")[:, : time.size]
    expected_contributors = np.sum(~inbag.astype(bool), axis=0)
    np.testing.assert_array_equal(with_oob.oob.contributor_count, expected_contributors)

    # Independently accumulate each OOB row's tree-specific leaf curves.
    grid = with_oob.time_grid
    survival_sum = np.zeros_like(with_oob.oob.survival)
    hazard_sum = np.zeros_like(with_oob.oob.cumulative_hazard)
    for tree_index, tree in enumerate(with_oob.trees):
        for row in np.flatnonzero(~inbag[tree_index, : time.size].astype(bool)):
            node = 0
            while tree.feature[node] >= 0:
                column = int(tree.feature[node])
                node = int(
                    tree.left[node] if x[row, column] <= tree.threshold[node] else tree.right[node]
                )
            offset = int(tree.event_offset[node])
            count = int(tree.event_count[node])
            if count:
                steps = slice(offset, offset + count)
                location = np.searchsorted(tree.event_time[steps], grid, side="right") - 1
                present = location >= 0
                curve = np.ones(grid.size)
                hazard = np.zeros(grid.size)
                curve[present] = np.exp(tree.log_survival[offset + location[present]])
                hazard[present] = tree.cumulative_hazard[offset + location[present]]
                survival_sum[row] += curve
                hazard_sum[row] += hazard
            else:
                survival_sum[row] += 1
    populated = expected_contributors > 0
    survival_sum[populated] /= expected_contributors[populated, None]
    hazard_sum[populated] /= expected_contributors[populated, None]
    np.testing.assert_allclose(with_oob.oob.survival[populated], survival_sum[populated])
    np.testing.assert_allclose(with_oob.oob.cumulative_hazard[populated], hazard_sum[populated])
    assert np.all(np.isnan(with_oob.oob.survival[~populated]))
    assert np.all(np.isnan(with_oob.oob.mortality[~populated]))

    profiles = np.array([[2.0, 1.0], [8.0, 2.0]])
    np.testing.assert_array_equal(
        predict_random_survival_forest(ordinary, profiles=profiles).survival,
        predict_random_survival_forest(with_oob, profiles=profiles).survival,
    )


def test_oob_no_contributors_and_concordance_ties() -> None:
    no_oob = fit_random_survival_forest(
        [1, 2, 3, 4],
        [1, 0, 1, 1],
        n_trees=2,
        sample_fraction=1.0,
        random_state=2,
        compute_oob=True,
    )
    assert no_oob.oob is not None
    np.testing.assert_array_equal(no_oob.oob.contributor_count, np.zeros(4, dtype=int))
    assert np.all(np.isnan(no_oob.oob.survival))
    assert no_oob.oob.comparable_pairs == 0
    assert np.isnan(no_oob.oob.concordance_error)
    with pytest.raises(ValueError, match="max_oob_cells"):
        fit_random_survival_forest(
            [1, 2, 3, 4],
            [1, 0, 1, 1],
            n_trees=2,
            sample_fraction=0.5,
            random_state=2,
            compute_oob=True,
            max_oob_cells=8,
        )
    with pytest.raises(ValueError, match="pairwise-concordance"):
        fit_random_survival_forest(
            [1, 2, 3, 4],
            [1, 0, 1, 1],
            n_trees=2,
            sample_fraction=0.5,
            random_state=2,
            compute_oob=True,
            max_oob_work=8,
        )

    # A censor exactly 1e-9 before an event is treated as tied, so the event is
    # ordered first; just beyond the tolerance it is not a comparable pair.
    error, pairs = _oob_concordance_error(
        np.array([0.0, 1e-9]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.ones(2, int)
    )
    assert pairs == 1
    assert error == 0.0
    error, pairs = _oob_concordance_error(
        np.array([0.0, np.nextafter(1e-9, np.inf)]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.ones(2, int),
    )
    assert pairs == 0
    assert np.isnan(error)

    # For tied failures the native rule awards full concordance only when risk
    # predictions are tied strictly within epsilon; at epsilon it is half.
    error, pairs = _oob_concordance_error(
        np.array([1.0, 1.0]), np.ones(2), np.array([0.0, 1e-9]), np.ones(2, int)
    )
    assert pairs == 1
    assert error == 0.5
    error, _ = _oob_concordance_error(
        np.array([1.0, 1.0]), np.ones(2), np.array([0.0, 0.5e-9]), np.ones(2, int)
    )
    assert error == 0.0
