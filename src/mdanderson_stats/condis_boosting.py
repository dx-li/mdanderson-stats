"""Gaussian gradient-boosting refinement for CondiS-X imputations.

The implementation follows the pinned gbm Gaussian path: a response-mean
initializer, residual trees grown best-first, and a 0.1 update applied once
per tree.  NumPy random streams are deterministic but do not reproduce R's
random-number generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .condis import CondiSImputation
from .condis_regularized import _design, _make_folds, _readonly_int


@dataclass(frozen=True)
class _BoostNode:
    feature: int = -1
    cut: float = 0.0
    left: int = -1
    right: int = -1
    missing: int = -1
    value: float = 0.0
    weight: int = 0


@dataclass(frozen=True)
class _BoostTree:
    nodes: tuple[_BoostNode, ...]


@dataclass(frozen=True)
class CondiSBoostingRefinement:
    """Selected boosting settings, fold scores, and the full-data refit."""

    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    tree_grid: NDArray[np.int64]
    depth_grid: NDArray[np.int64]
    mean_rmse: FloatArray
    sd_rmse: FloatArray
    fold_rmse: FloatArray
    best_index: int
    best_tree_count: int
    best_depth: int
    fold_ids: NDArray[np.int64]
    trees: tuple[_BoostTree, ...]
    initial_mean: float
    shrinkage: float
    random_state: int | None
    enforce_censoring: bool


def _stable_mean(values: FloatArray) -> float:
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0.0 else scale * float(np.mean(values / scale))


def _sample_rows(uniforms: FloatArray, size: int) -> NDArray[np.int64]:
    """Reproduce gbm's sequential without-replacement draw consumption."""
    remaining = size
    chosen = np.zeros(uniforms.size, dtype=np.bool_)
    for row, draw in enumerate(uniforms):
        rows_left = uniforms.size - row
        if draw * rows_left < remaining:
            chosen[row] = True
            remaining -= 1
    if remaining != 0:
        raise ArithmeticError("uniform stream did not produce the requested bag size")
    return np.flatnonzero(chosen)


def _best_split(
    rows: NDArray[np.int64],
    residual: FloatArray,
    x: FloatArray,
    feature_orders: tuple[NDArray[np.int64], ...],
    residual_scale: float,
    min_node: int,
) -> tuple[float, int, float, NDArray[np.int64], NDArray[np.int64]] | None:
    best_gain = 0.0
    best: tuple[float, int, float, NDArray[np.int64], NDArray[np.int64]] | None = None
    row_mask = np.zeros(x.shape[0], dtype=np.bool_)
    row_mask[rows] = True
    for feature, order in enumerate(feature_orders):
        ordered_rows = order[row_mask[order]]
        if ordered_rows.size < 2 * min_node:
            continue
        values = x[ordered_rows, feature]
        z = residual[ordered_rows] / residual_scale if residual_scale else residual[ordered_rows]
        cumulative = np.cumsum(z, dtype=np.float64)
        split_positions = np.flatnonzero(values[:-1] < values[1:]) + 1
        valid = split_positions[
            (split_positions >= min_node) & (ordered_rows.size - split_positions >= min_node)
        ]
        if valid.size == 0:
            continue
        left_sum = cumulative[valid - 1]
        total_sum = cumulative[-1]
        left_n = valid.astype(np.float64)
        right_n = ordered_rows.size - left_n
        # This is the Gaussian weighted-mean split improvement, scaled by one
        # tree-wide residual factor so candidate ordering cannot overflow.
        gain = (
            left_n
            * right_n
            * np.square(left_sum / left_n - (total_sum - left_sum) / right_n)
            / (left_n + right_n)
        )
        position_index = int(np.argmax(gain))
        candidate_gain = float(gain[position_index])
        if candidate_gain > best_gain:
            position = int(valid[position_index])
            lower = float(values[position - 1])
            upper = float(values[position])
            cut = lower + (upper - lower) * 0.5
            if not np.isfinite(cut):
                cut = lower * 0.5 + upper * 0.5
            if cut <= lower:
                cut = upper
            left = ordered_rows[:position]
            right = ordered_rows[position:]
            best_gain = candidate_gain
            best = (candidate_gain, feature, cut, left.copy(), right.copy())
    return best


def _predict_tree(tree: _BoostTree, x: FloatArray) -> FloatArray:
    output = np.empty(x.shape[0], dtype=np.float64)
    for row in range(x.shape[0]):
        node_index = 0
        while True:
            node = tree.nodes[node_index]
            if node.feature < 0:
                output[row] = node.value
                break
            value = x[row, node.feature]
            node_index = node.left if value < node.cut else node.right
    return output


def _build_tree(
    x: FloatArray,
    residual: FloatArray,
    bag: NDArray[np.int64],
    feature_orders: tuple[NDArray[np.int64], ...],
    depth: int,
    min_node: int,
) -> _BoostTree:
    residual_scale = float(np.max(np.abs(residual[bag])))
    root_value = _stable_mean(residual[bag])
    nodes: list[_BoostNode] = [_BoostNode(value=root_value, weight=int(bag.size))]
    leaves: list[tuple[int, NDArray[np.int64]]] = [(0, bag)]
    for _ in range(depth):
        selected_position = -1
        selected_split = None
        best_gain = 0.0
        for position, (node_index, rows) in enumerate(leaves):
            split = _best_split(rows, residual, x, feature_orders, residual_scale, min_node)
            if split is not None and split[0] > best_gain:
                best_gain = split[0]
                selected_position = position
                selected_split = split
        if selected_split is None or selected_position < 0:
            break
        node_index, parent_rows = leaves[selected_position]
        _, feature, cut, left_rows, right_rows = selected_split
        left_value = _stable_mean(residual[left_rows])
        right_value = _stable_mean(residual[right_rows])
        left_index = len(nodes)
        right_index = left_index + 1
        missing_index = left_index + 2
        parent_value = nodes[node_index].value
        nodes.extend(
            (
                _BoostNode(value=left_value, weight=int(left_rows.size)),
                _BoostNode(value=right_value, weight=int(right_rows.size)),
                # Native exported trees store the parent value/weight in an
                # empty missing branch; finite CondiS designs never traverse it.
                _BoostNode(value=parent_value, weight=int(parent_rows.size)),
            )
        )
        nodes[node_index] = _BoostNode(
            feature=feature,
            cut=cut,
            left=left_index,
            right=right_index,
            missing=missing_index,
            value=parent_value,
            weight=int(parent_rows.size),
        )
        leaves[selected_position : selected_position + 1] = [
            (left_index, left_rows),
            (right_index, right_rows),
            (missing_index, np.empty(0, dtype=np.int64)),
        ]
    return _BoostTree(tuple(nodes))


def _fit_boosted_path(
    x: FloatArray,
    y: FloatArray,
    uniforms: ArrayLike | np.random.Generator,
    *,
    n_trees: int,
    depth: int,
    shrinkage: float = 0.1,
    min_node_observations: int = 10,
) -> tuple[tuple[_BoostTree, ...], FloatArray, FloatArray]:
    """Fit a deterministic path from row-by-tree uniforms; useful for replay."""
    if x.ndim != 2 or y.shape != (x.shape[0],) or not np.isfinite(x).all():
        raise ValueError("x and y must be finite arrays with matching rows")
    stream: FloatArray | None
    random_stream: np.random.Generator | None
    if isinstance(uniforms, np.random.Generator):
        stream = None
        random_stream = uniforms
    else:
        random_stream = None
        stream = finite(uniforms, "uniforms")
        if stream.shape != (x.shape[0], n_trees) or np.any((stream < 0.0) | (stream >= 1.0)):
            raise ValueError("uniforms must have shape (n_rows, n_trees) and lie in [0, 1)")
    bag_size = int(np.floor(0.5 * x.shape[0]))
    if x.shape[0] * 0.5 <= 2 * min_node_observations + 1:
        raise ValueError("gbm default bag-fraction guard requires at least 43 training rows")
    feature_orders = tuple(np.argsort(x[:, j], kind="stable") for j in range(x.shape[1]))
    fitted = np.full(x.shape[0], _stable_mean(y), dtype=np.float64)
    initial = float(fitted[0])
    trees: list[_BoostTree] = []
    for tree_index in range(n_trees):
        residual = y - fitted
        if not np.isfinite(residual).all():
            raise ArithmeticError("boosting residuals became non-finite")
        draw: FloatArray
        if random_stream is not None:
            draw = np.asarray(random_stream.random(x.shape[0]), dtype=np.float64)
        else:
            assert stream is not None
            draw = stream[:, tree_index]
        bag = _sample_rows(draw, bag_size)
        tree = _build_tree(x, residual, bag, feature_orders, depth, min_node_observations)
        fitted += shrinkage * _predict_tree(tree, x)
        if not np.isfinite(fitted).all():
            raise ArithmeticError("boosting predictions became non-finite")
        trees.append(tree)
    return tuple(trees), fitted, np.asarray([initial], dtype=np.float64)


def condis_boosting_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    folds: int = 10,
    repeats: int = 1,
    random_state: int | None = 0,
    fold_ids: ArrayLike | None = None,
    tree_grid: ArrayLike = (50, 100, 150),
    depth_grid: ArrayLike = (1, 2, 3),
    enforce_censoring: bool = False,
) -> CondiSBoostingRefinement:
    """Tune Gaussian boosting by repeated fold RMSE, then refit the winner.

    Each fold/depth fit grows one 150-tree path and scores the requested
    prefixes. Fold labels and the random seed are explicit; NumPy draws do not
    claim R RNG parity with gbm.
    """
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    x, y = _design(imputation, covariates)
    n = y.size
    trees = np.asarray(tree_grid)
    depths = np.asarray(depth_grid)
    if trees.ndim != 1 or trees.size == 0 or trees.dtype.kind not in "iu" or trees.size > 10:
        raise ValueError("tree_grid must be a nonempty integer vector of at most 10 values")
    if depths.ndim != 1 or depths.size == 0 or depths.dtype.kind not in "iu" or depths.size > 10:
        raise ValueError("depth_grid must be a nonempty integer vector of at most 10 values")
    trees = np.unique(trees.astype(np.int64))
    depths = np.unique(depths.astype(np.int64))
    if np.any(trees < 1) or np.any(trees > 150) or np.any(depths < 1) or np.any(depths > 3):
        raise ValueError("tree counts must be 1..150 and depths 1..3")
    if not np.array_equal(trees, np.asarray(tree_grid)) or not np.array_equal(
        depths, np.asarray(depth_grid)
    ):
        # Unique sorted grids are required for unambiguous caret tie ordering.
        raise ValueError("tree_grid and depth_grid must be unique and increasing")
    assignments = _make_folds(n, y, folds, repeats, random_state, fold_ids)
    fold_count = min(int(folds), n)
    for rep in range(repeats):
        for fold in range(int(fold_count)):
            n_train = int(np.count_nonzero(assignments[rep] != fold))
            if n_train * 0.5 <= 21:
                raise ValueError("each boosting CV training fold must contain at least 43 rows")
    work = (
        n * x.shape[1] * int(np.max(trees)) * int(np.sum(depths)) * (repeats * int(fold_count) + 1)
    )
    if work > 250_000_000:
        raise ValueError("boosting work exceeds the 250 million row-feature-split budget")
    rng = np.random.default_rng(random_state)
    score_grid = [(int(ntree), int(depth)) for ntree in trees for depth in depths]
    scores = np.empty((repeats, int(fold_count), len(score_grid)), dtype=np.float64)
    max_trees = int(np.max(trees))
    for rep in range(repeats):
        for fold in range(int(fold_count)):
            test_mask = assignments[rep] == fold
            x_train, y_train = x[~test_mask], y[~test_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            for depth in depths:
                fitted_trees, _, initial = _fit_boosted_path(
                    x_train, y_train, rng, n_trees=max_trees, depth=int(depth)
                )
                for grid_index, (tree_count, grid_depth) in enumerate(score_grid):
                    if grid_depth != depth:
                        continue
                    prediction = _predict_path_prefix(
                        x_test, fitted_trees, tree_count, float(initial[0])
                    )
                    scale = max(float(np.max(np.abs(y_test))), float(np.max(np.abs(prediction))))
                    scale = scale if scale else 1.0
                    residual = y_test / scale - prediction / scale
                    scores[rep, fold, grid_index] = scale * float(
                        np.sqrt(np.mean(residual * residual))
                    )
    if not np.isfinite(scores).all():
        raise ArithmeticError("boosting cross-validation produced non-finite RMSE")
    flat = scores.reshape((-1, len(score_grid)))
    score_scale = float(np.max(np.abs(flat)))
    normalized = flat / score_scale if score_scale else flat
    means = score_scale * normalized.mean(axis=0)
    sds = (
        score_scale * normalized.std(axis=0, ddof=1)
        if flat.shape[0] > 1
        else np.zeros(len(score_grid))
    )
    best = int(np.argmin(means))
    best_count, best_depth = score_grid[best]
    final_trees, fitted, initial_array = _fit_boosted_path(
        x, y, rng, n_trees=best_count, depth=best_depth
    )
    if not np.isfinite(fitted).all():
        raise ArithmeticError("selected boosting refit produced non-finite predictions")
    censored = imputation.status == 0
    below = censored & (fitted < imputation.observed_time)
    above = censored & (fitted > imputation.horizon)
    refined = np.where(censored, fitted, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    return CondiSBoostingRefinement(
        _owned(fitted),
        _owned(refined),
        _owned(below),
        _owned(above),
        _readonly_int(trees),
        _readonly_int(depths),
        _owned(means),
        _owned(sds),
        _owned(scores),
        best,
        best_count,
        best_depth,
        _readonly_int(assignments),
        final_trees,
        float(initial_array[0]),
        0.1,
        None if random_state is None else int(random_state),
        enforce_censoring,
    )


def _predict_path_prefix(
    x_test: FloatArray, trees: tuple[_BoostTree, ...], tree_count: int, initial: float
) -> FloatArray:
    prediction = np.full(x_test.shape[0], initial, dtype=np.float64)
    for tree in trees[:tree_count]:
        prediction += 0.1 * _predict_tree(tree, x_test)
    return prediction
