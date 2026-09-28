"""Explicit permutation variable importance for numeric survival forests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .random_survival_forest import (
    RandomSurvivalForestFit,
    _forest_data,
    _forest_fingerprint,
    _integer,
    _oob_concordance_error,
    _PackedTree,
)

_MAX_IMPORTANCE_WORK = 100_000_000
_MAX_IMPORTANCE_CELLS = 2_000_000


@dataclass(frozen=True)
class RandomSurvivalForestPermutationImportance:
    """Blockwise permutation increase in OOB concordance error.

    ``importance`` is the mean of valid block differences. The native source
    uses ``floor(n_trees / block_size)`` complete blocks and ignores the tail;
    those zero-based tree indices are returned in ``ignored_tree_indices``.
    """

    feature_indices: np.ndarray
    block_size: int
    block_count: int
    ignored_tree_indices: np.ndarray
    baseline_error: FloatArray
    perturbed_error: FloatArray
    block_importance: FloatArray
    valid_block_count: np.ndarray
    importance: FloatArray
    random_state: int | None


def _route_hazard(
    tree: _PackedTree,
    profile: FloatArray,
    time_grid: FloatArray,
    *,
    override_column: int = -1,
    override_value: float = 0.0,
) -> float:
    node = 0
    while tree.feature[node] >= 0:
        column = int(tree.feature[node])
        value = override_value if column == override_column else profile[column]
        node = int(tree.left[node] if value <= tree.threshold[node] else tree.right[node])
    count = int(tree.event_count[node])
    if count == 0 or time_grid.size == 0:
        return 0.0
    offset = int(tree.event_offset[node])
    steps = slice(offset, offset + count)
    positions = np.searchsorted(tree.event_time[steps], time_grid, side="right") - 1
    present = positions >= 0
    return float(tree.cumulative_hazard[offset + positions[present]].sum())


def _feature_selection(value: ArrayLike | None, count: int) -> np.ndarray:
    if value is None:
        return np.arange(count, dtype=np.int64)
    if isinstance(value, np.ndarray):
        if value.size > count:
            raise ValueError("feature_indices must contain unique fitted column indices")
        if np.iscomplexobj(value):
            raise ValueError("feature_indices must be real integers")
        raw = value
    elif isinstance(value, (list, tuple)):
        if len(value) > count:
            raise ValueError("feature_indices must contain unique fitted column indices")
        if any(
            isinstance(item, (bool, np.bool_, list, tuple, np.ndarray))
            or not isinstance(item, (int, np.integer))
            for item in value
        ):
            raise ValueError("feature_indices must be a flat sequence of integer indices")
        raw = np.asarray(value)
    else:
        raise ValueError("feature_indices must be a bounded one-dimensional integer sequence")
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError("feature_indices must be a nonempty integer vector")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= count) or np.unique(result).size != result.size:
        raise ValueError("feature_indices must be unique fitted column indices")
    return result


def _readonly_int(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.asarray(value, dtype=np.int64).tobytes(), dtype=np.int64)


def permutation_random_survival_forest_importance(
    fit: RandomSurvivalForestFit,
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    feature_indices: ArrayLike | None = None,
    block_size: int | None = None,
    rng: np.random.Generator | None = None,
    random_state: int | None = None,
    max_work: int = _MAX_IMPORTANCE_WORK,
) -> RandomSurvivalForestPermutationImportance:
    """Compute explicit permutation VIMP from a fit made with ``compute_oob``.

    Each selected feature is independently permuted within each tree's OOB
    rows, then those rows are rerouted. A block score is the perturbed minus
    faithful block ``1-C`` error. The routine is a permutation option and does
    not implement randomForestSRC's default anti-split importance.
    """
    if fit.inbag_membership is None or fit.oob is None or fit.training_fingerprint is None:
        raise ValueError("fit must be created with compute_oob=True for permutation importance")
    if (
        len(fit.trees) != fit.n_trees
        or fit.inbag_membership.ndim != 2
        or fit.inbag_membership.shape[0] != fit.n_trees
        or fit.inbag_membership.shape[1] * 8 < fit.oob.contributor_count.size
        or fit.inbag_membership.shape[1] * 8 - fit.oob.contributor_count.size >= 8
    ):
        raise ValueError("fit contains inconsistent packed OOB membership")
    if (rng is None) == (random_state is None):
        raise ValueError("provide exactly one of rng or random_state")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if random_state is not None:
        random_state = _integer(random_state, "random_state", 0, np.iinfo(np.int32).max)
        generator = np.random.default_rng(random_state)
    else:
        assert rng is not None
        generator = rng
    work_limit = _integer(max_work, "max_work", 1, _MAX_IMPORTANCE_WORK)
    t, e, x = _forest_data(time, event, covariates)
    if x.shape[1] != fit.covariate_count:
        raise ValueError("training covariate count does not match the OOB fit")
    if _forest_fingerprint(t, e, x) != fit.training_fingerprint:
        raise ValueError("training data values or row order do not match the OOB fit")
    selected = _feature_selection(feature_indices, fit.covariate_count)
    if selected.size == 0:
        raise ValueError("permutation importance requires at least one fitted covariate")
    if block_size is None:
        size = fit.n_trees
    else:
        size = _integer(block_size, "block_size", 1, fit.n_trees)
    block_count = fit.n_trees // size
    used_trees = block_count * size
    ignored = np.arange(used_trees, fit.n_trees, dtype=np.int64)
    output_cells = selected.size * block_count
    if 2 * output_cells + block_count + selected.size + ignored.size > _MAX_IMPORTANCE_CELLS:
        raise ValueError("permutation importance outputs exceed the combined cell budget")
    rows = t.size
    oob_by_tree = np.empty(used_trees, dtype=np.int64)
    for tree_index in range(used_trees):
        packed = fit.inbag_membership[tree_index]
        oob_by_tree[tree_index] = rows - int(np.unpackbits(packed, bitorder="little")[:rows].sum())
    route_rows = int(oob_by_tree.sum())
    route_work = selected.size * route_rows * (fit.time_grid.size + fit.max_depth) * 2
    pair_work = (selected.size + 1) * block_count * rows * rows
    if route_work + pair_work > work_limit:
        raise ValueError("permutation routing/concordance work exceeds max_work")

    baseline_error = np.full(block_count, np.nan, dtype=np.float64)
    perturbed_error = np.full((selected.size, block_count), np.nan, dtype=np.float64)
    block_importance = np.full_like(perturbed_error, np.nan)
    for block in range(block_count):
        first_tree = block * size
        last_tree = first_tree + size
        baseline_sum = np.zeros(rows, dtype=np.float64)
        contributors = np.zeros(rows, dtype=np.int64)
        for tree_index in range(first_tree, last_tree):
            rows_inbag = np.unpackbits(fit.inbag_membership[tree_index], bitorder="little")[
                :rows
            ].astype(bool)
            oob_rows = np.flatnonzero(~rows_inbag)
            tree = fit.trees[tree_index]
            for row in oob_rows:
                baseline_sum[row] += _route_hazard(tree, x[row], fit.time_grid)
                contributors[row] += 1
        baseline_mortality = np.full(rows, np.nan, dtype=np.float64)
        present = contributors > 0
        baseline_mortality[present] = baseline_sum[present] / contributors[present]
        baseline_error[block], _ = _oob_concordance_error(t, e, baseline_mortality, contributors)
        for feature_position, feature in enumerate(selected):
            perturbed_sum = np.zeros(rows, dtype=np.float64)
            for tree_index in range(first_tree, last_tree):
                rows_inbag = np.unpackbits(fit.inbag_membership[tree_index], bitorder="little")[
                    :rows
                ].astype(bool)
                oob_rows = np.flatnonzero(~rows_inbag)
                if oob_rows.size == 0:
                    continue
                values = x[oob_rows, int(feature)]
                shuffled = values[generator.permutation(oob_rows.size)]
                tree = fit.trees[tree_index]
                for row, value in zip(oob_rows, shuffled, strict=True):
                    perturbed_sum[row] += _route_hazard(
                        tree,
                        x[row],
                        fit.time_grid,
                        override_column=int(feature),
                        override_value=float(value),
                    )
            perturbed_mortality = np.full(rows, np.nan, dtype=np.float64)
            perturbed_mortality[present] = perturbed_sum[present] / contributors[present]
            perturbed_error[feature_position, block], _ = _oob_concordance_error(
                t, e, perturbed_mortality, contributors
            )
            if np.isfinite(baseline_error[block]) and np.isfinite(
                perturbed_error[feature_position, block]
            ):
                block_importance[feature_position, block] = (
                    perturbed_error[feature_position, block] - baseline_error[block]
                )
    valid = np.isfinite(block_importance)
    valid_count = valid.sum(axis=1).astype(np.int64)
    importance = np.full(selected.size, np.nan, dtype=np.float64)
    found = valid_count > 0
    importance[found] = np.nansum(block_importance[found], axis=1) / valid_count[found]
    return RandomSurvivalForestPermutationImportance(
        _readonly_int(selected),
        size,
        block_count,
        _readonly_int(ignored),
        _freeze(baseline_error),
        _freeze(perturbed_error),
        _freeze(block_importance),
        _readonly_int(valid_count),
        _freeze(importance),
        random_state,
    )
