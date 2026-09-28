"""Numeric regression-forest refinement for CondiS-X.

This is an independent NumPy implementation of the numeric ``randomForest``
regression tree path. It keeps bootstrap multiplicities, random feature/tie
draws, native zero-gain candidate splits, and all-tree prediction. NumPy seeds
do not reproduce R's RNG; an explicit ``uniform_draws`` stream enables
controlled comparisons, but tied-sort and degenerate-partition behavior can
diverge from native R after a path-specific tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .condis import CondiSImputation
from .condis_regularized import _design, _make_folds

_MAX_ROWS = 5_000
_MAX_FEATURES = 100
_MAX_DESIGN_CELLS = 500_000
_MAX_TREES = 2_000
_MAX_WEIGHT_CELLS = 2_000_000
_MAX_SPLIT_WORK = 100_000_000
_MAX_NODES = 2_000_000
_MAX_STREAM = 10_000_000


@dataclass(frozen=True)
class _RegressionTree:
    feature: NDArray[np.int32]  # 0-based split feature; -1 for terminal nodes
    threshold: FloatArray
    left: NDArray[np.int32]
    right: NDArray[np.int32]
    node_mean: FloatArray  # on centered, optionally range-scaled response
    status: NDArray[np.int32]  # -1 interior, -3 terminal


@dataclass(frozen=True)
class CondiSForestFit:
    """Packed regression forest; predictions average all trees, not OOB trees."""

    trees: tuple[_RegressionTree, ...]
    response_center: float
    response_scale: float
    predictor_count: int
    mtry: int
    nodesize: int
    n_trees: int
    random_state: int | None
    uniform_draws_used: int
    node_count: int
    split_work: int
    inbag_counts: NDArray[np.int32] | None


@dataclass(frozen=True)
class CondiSForestRefinement:
    """Single-setting fold scores and selected full-data forest refinement."""

    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    mtry: int
    fold_rmse: FloatArray
    mean_rmse: float
    sd_rmse: float
    fold_ids: NDArray[np.int64]
    fold_node_count: NDArray[np.int64]
    fold_split_work: NDArray[np.int64]
    random_state: int | None
    enforce_censoring: bool
    fit: CondiSForestFit


class _UniformSource:
    def __init__(self, draws: ArrayLike | None, seed: int | None):
        self.values: FloatArray | None = None
        self.position = 0
        self.rng = None
        if draws is not None:
            if np.iscomplexobj(draws):
                raise ValueError("uniform_draws must be real")
            raw = np.asarray(draws)
            if raw.ndim != 1 or raw.size > _MAX_STREAM:
                raise ValueError(f"uniform_draws must be a vector of at most {_MAX_STREAM}")
            values = finite(draws, "uniform_draws")
            if np.any((values < 0.0) | (values >= 1.0)):
                raise ValueError("uniform_draws must lie in [0, 1)")
            self.values = values
        else:
            self.rng = np.random.default_rng(seed)

    def take(self) -> float:
        if self.values is not None:
            if self.position >= self.values.size:
                raise ValueError("uniform_draws ended before forest construction completed")
            value = float(self.values[self.position])
            self.position += 1
            return value
        assert self.rng is not None
        self.position += 1
        return float(self.rng.random())


def _int(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return result


def _stable_mean(values: FloatArray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(values / scale))
    if not np.isfinite(result):
        raise ArithmeticError("response mean is not representable")
    return result


def _node_mean(values: FloatArray) -> float:
    mean = 0.0
    for i, value in enumerate(values):
        mean = (i * mean + float(value)) / (i + 1)
    return mean


def _midpoint(lower: float, upper: float) -> float:
    point = lower / 2.0 + upper / 2.0
    # RF routes values <= the split to the left.  When adjacent floats have no
    # interior midpoint, choosing ``upper`` would make both go left.
    if not lower < point < upper:
        point = lower
    return point


def _grow_tree(
    x: FloatArray,
    response: FloatArray,
    bootstrap: np.ndarray,
    *,
    mtry: int,
    nodesize: int,
    stream: _UniformSource,
    max_nodes_remaining: int,
    max_split_work: int,
) -> tuple[_RegressionTree, int, int]:
    features = [-1]
    thresholds = [0.0]
    left = [-1]
    right = [-1]
    means = [_node_mean(response[bootstrap])]
    status = [-1]  # Root is always offered a split, even at nodesize or smaller.
    node_rows: list[np.ndarray | None] = [bootstrap]
    work = 0
    node = 0
    while node < len(features):
        rows = node_rows[node]
        if rows is None:
            node += 1
            continue
        node_rows[node] = None
        best_feature = -1
        best_threshold = 0.0
        best_global = 0.0
        # randomForest keeps this threshold across predictor candidates.  In
        # an exact cross-variable gain tie its native code can pair the
        # selected variable with the last threshold tie selected below.
        value_at_best_split = 0.0
        tie_variable = 1
        next_features = list(range(x.shape[1]))
        last = len(next_features) - 1
        if rows.size > 1:
            parent_mean = means[node]
            parent_sum = float(rows.size * parent_mean)
            for _ in range(min(mtry, x.shape[1])):
                index = int(stream.take() * (last + 1))
                variable = next_features[index]
                next_features[index] = next_features[last]
                last -= 1
                values = x[rows, variable]
                order = np.argsort(values, kind="quicksort")
                sorted_values = values[order]
                if sorted_values[0] >= sorted_values[-1]:
                    continue
                sorted_response = response[rows[order]]
                sum_left = 0.0
                sum_right = parent_sum
                best_within = 0.0
                tie_value = 1
                parent_term = parent_sum * parent_sum / rows.size
                for j in range(rows.size - 1):
                    y_value = float(sorted_response[j])
                    sum_left += y_value
                    sum_right -= y_value
                    if sorted_values[j] < sorted_values[j + 1]:
                        work += 1
                        if work > max_split_work:
                            raise ValueError("forest exceeds bounded split work")
                        left_n = j + 1
                        right_n = rows.size - left_n
                        criterion = (
                            sum_left * sum_left / left_n
                            + sum_right * sum_right / right_n
                            - parent_term
                        )
                        if not np.isfinite(criterion):
                            raise ArithmeticError("regression split criterion is not representable")
                        if criterion > best_within:
                            value_at_best_split = _midpoint(
                                float(sorted_values[j]), float(sorted_values[j + 1])
                            )
                            best_within = criterion
                            tie_value = 1
                        if criterion == best_within:
                            tie_value += 1
                            if stream.take() < 1.0 / tie_value:
                                value_at_best_split = _midpoint(
                                    float(sorted_values[j]), float(sorted_values[j + 1])
                                )
                                best_within = criterion
                if best_within > best_global:
                    best_threshold = value_at_best_split
                    best_feature = variable
                    best_global = best_within
                    tie_variable = 1
                if best_within == best_global:
                    tie_variable += 1
                    if stream.take() < 1.0 / tie_variable:
                        best_threshold = value_at_best_split
                        best_feature = variable
                        best_global = best_within
        if best_feature < 0:
            status[node] = -3
            node += 1
            continue
        goes_left = x[rows, best_feature] <= best_threshold
        left_rows, right_rows = rows[goes_left], rows[~goes_left]
        if left_rows.size == 0 or right_rows.size == 0:
            # Native RF forces a non-empty partition even for a degenerate
            # zero-gain split.  This bounded Python implementation keeps the
            # node terminal instead of inventing an ordering-dependent split.
            status[node] = -3
            node += 1
            continue
        left_id, right_id = len(features), len(features) + 1
        features[node], thresholds[node], left[node], right[node] = (
            best_feature,
            best_threshold,
            left_id,
            right_id,
        )
        status[node] = -1
        for child_rows in (left_rows, right_rows):
            child_mean = _node_mean(response[child_rows])
            features.append(-1)
            thresholds.append(0.0)
            left.append(-1)
            right.append(-1)
            means.append(child_mean)
            status.append(-1 if child_rows.size > nodesize else -3)
            node_rows.append(child_rows if child_rows.size > nodesize else None)
        if len(features) > max_nodes_remaining:
            raise ValueError("forest exceeds max_nodes budget")
        node += 1
    tree = _RegressionTree(
        _freeze_int(np.asarray(features)),
        _owned(np.asarray(thresholds, dtype=np.float64)),
        _freeze_int(np.asarray(left)),
        _freeze_int(np.asarray(right)),
        _owned(np.asarray(means, dtype=np.float64)),
        _freeze_int(np.asarray(status)),
    )
    return tree, len(features), work


def fit_condis_forest(
    predictors: ArrayLike,
    target: ArrayLike,
    *,
    n_trees: int = 500,
    mtry: int | None = None,
    nodesize: int = 5,
    random_state: int | None = 0,
    uniform_draws: ArrayLike | None = None,
    keep_inbag: bool = False,
) -> CondiSForestFit:
    """Fit numeric CART regression trees with bootstrap and random mtry.

    ``predictors`` should already include the status indicator as the first
    column for CondiS-X. A standalone omitted ``mtry`` uses rounded square root
    of the supplied predictor count; the CondiS wrapper instead derives it
    from the original covariate count, excluding status.
    """
    if np.iscomplexobj(predictors) or np.iscomplexobj(target):
        raise ValueError("predictors and target must be real")
    raw_x, raw_y = np.asarray(predictors), np.asarray(target)
    if (
        raw_x.ndim != 2
        or raw_y.ndim != 1
        or raw_x.size > _MAX_DESIGN_CELLS
        or raw_y.size > _MAX_ROWS
    ):
        raise ValueError("forest data exceed the bounded row/design dimensions")
    x, y = finite(predictors, "predictors"), finite(target, "target")
    if (
        x.ndim != 2
        or y.ndim != 1
        or x.shape[0] != y.size
        or not 1 <= y.size <= _MAX_ROWS
        or not 1 <= x.shape[1] <= _MAX_FEATURES
        or x.size > _MAX_DESIGN_CELLS
    ):
        raise ValueError("predictors need 1..100 columns, 1..5000 rows and <=500,000 cells")
    tree_count = _int(n_trees, "n_trees", 1, _MAX_TREES)
    leaf_size = _int(nodesize, "nodesize", 1, _MAX_ROWS)
    feature_count = (
        int(np.rint(np.sqrt(x.shape[1]))) if mtry is None else _int(mtry, "mtry", 1, x.shape[1])
    )
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be None or a nonnegative integer")
    if not isinstance(keep_inbag, bool):
        raise ValueError("keep_inbag must be boolean")
    if uniform_draws is not None and random_state not in (None, 0):
        raise ValueError("random_state is ignored for explicit uniform_draws; pass None or 0")
    if keep_inbag and y.size * tree_count > _MAX_WEIGHT_CELLS:
        raise ValueError("in-bag output exceeds the 2 million cell budget")
    # n trees × bootstrap n is also the dominant obligatory forest work bound.
    sampled_rows = tree_count * y.size
    if sampled_rows > _MAX_WEIGHT_CELLS:
        raise ValueError("forest exceeds the 2 million sampled-row budget")
    if tree_count * (2 * y.size - 1) > _MAX_NODES:
        raise ValueError("forest exceeds the 2 million packed-node budget")
    source = _UniformSource(uniform_draws, None if uniform_draws is not None else random_state)
    response_center = _stable_mean(y)
    centered = y - response_center
    if not np.isfinite(centered).all():
        raise ArithmeticError("centered response is not representable; rescale time units")
    response_scale = float(np.max(np.abs(centered)))
    if response_scale > 1e150 or (0.0 < response_scale < 1e-150):
        centered = centered / response_scale
    else:
        response_scale = 1.0
    all_trees: list[_RegressionTree] = []
    total_nodes = total_work = 0
    inbag = np.zeros((y.size, tree_count), dtype=np.int32) if keep_inbag else None
    for tree_index in range(tree_count):
        bootstrap: NDArray[np.int64] = np.fromiter(
            (int(source.take() * y.size) for _ in range(y.size)), dtype=np.int64, count=y.size
        )
        if inbag is not None:
            inbag[:, tree_index] = np.bincount(bootstrap, minlength=y.size)
        tree, nodes, work = _grow_tree(
            x,
            centered,
            bootstrap,
            mtry=feature_count,
            nodesize=leaf_size,
            stream=source,
            max_nodes_remaining=_MAX_NODES - total_nodes,
            max_split_work=_MAX_SPLIT_WORK - total_work,
        )
        total_nodes += nodes
        total_work += work
        all_trees.append(tree)
    return CondiSForestFit(
        tuple(all_trees),
        response_center,
        response_scale,
        x.shape[1],
        feature_count,
        leaf_size,
        tree_count,
        None if random_state is None else int(random_state),
        source.position,
        total_nodes,
        total_work,
        None if inbag is None else _freeze_int(inbag),
    )


def _predict_one(tree: _RegressionTree, x: FloatArray) -> FloatArray:
    prediction = np.empty(x.shape[0], dtype=np.float64)
    for row in range(x.shape[0]):
        node = 0
        while tree.status[node] == -1:
            node = int(
                tree.left[node]
                if x[row, tree.feature[node]] <= tree.threshold[node]
                else tree.right[node]
            )
        prediction[row] = tree.node_mean[node]
    return prediction


def predict_condis_forest(
    fit: CondiSForestFit,
    predictors: ArrayLike,
    *,
    individual: bool = False,
) -> FloatArray:
    """Predict with all trees; optionally return bounded row-by-tree values."""
    if not isinstance(fit, CondiSForestFit):
        raise TypeError("fit must be CondiSForestFit")
    if np.iscomplexobj(predictors):
        raise ValueError("predictors must be real")
    raw = np.asarray(predictors)
    if raw.ndim not in (1, 2) or raw.size > _MAX_DESIGN_CELLS:
        raise ValueError("predictors exceed the bounded prediction dimensions")
    x = finite(predictors, "predictors")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] != fit.predictor_count or x.size > _MAX_DESIGN_CELLS:
        raise ValueError("predictors must have the fitted columns and bounded dimensions")
    if not isinstance(individual, bool):
        raise ValueError("individual must be boolean")
    if individual and x.shape[0] * fit.n_trees > _MAX_WEIGHT_CELLS:
        raise ValueError("individual tree predictions exceed the 2 million cell budget")
    total = np.zeros(x.shape[0], dtype=np.float64)
    by_tree = np.empty((x.shape[0], fit.n_trees), dtype=np.float64) if individual else None
    for i, tree in enumerate(fit.trees):
        values = _predict_one(tree, x)
        total += values
        if by_tree is not None:
            by_tree[:, i] = values
    prediction = fit.response_center + fit.response_scale * (total / fit.n_trees)
    if not np.isfinite(prediction).all():
        raise ArithmeticError("forest prediction is not representable; rescale time units")
    if by_tree is not None:
        by_tree = fit.response_center + fit.response_scale * by_tree
        return _owned(by_tree)
    return _owned(prediction)


def condis_forest_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    folds: int = 10,
    fold_ids: ArrayLike | None = None,
    random_state: int | None = 0,
    n_trees: int = 500,
    enforce_censoring: bool = False,
) -> CondiSForestRefinement:
    """Cross-validate and refit the single CondiS random-forest setting."""
    if not isinstance(imputation, CondiSImputation):
        raise TypeError("imputation must be CondiSImputation")
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    if imputation.imputed_time.size > _MAX_ROWS:
        raise ValueError(f"random-forest refinement is limited to {_MAX_ROWS} rows")
    if np.iscomplexobj(covariates):
        raise ValueError("covariates must be real")
    raw_covariates = np.asarray(covariates)
    if raw_covariates.ndim != 2 or raw_covariates.shape[0] != imputation.imputed_time.size:
        raise ValueError("covariates must be a matrix with one row per imputed observation")
    if (
        raw_covariates.shape[1] > _MAX_FEATURES - 1
        or raw_covariates.size + raw_covariates.shape[0] > _MAX_DESIGN_CELLS
    ):
        raise ValueError("CondiS forest design exceeds the bounded predictor dimensions")
    design, y = _design(imputation, covariates)
    if design.shape[1] > _MAX_FEATURES or not np.isfinite(y).all():
        raise ValueError("CondiS forest requires at most 99 covariates and finite imputed times")
    if (
        isinstance(folds, (bool, np.bool_))
        or not isinstance(folds, (int, np.integer))
        or not 2 <= folds <= 100
    ):
        raise ValueError("folds must be an integer in [2, 100]")
    if y.size < 2:
        raise ValueError("cross-validation requires at least two observations")
    fold_count = min(int(folds), y.size)
    assignments = _make_folds(y.size, y, fold_count, 1, random_state, fold_ids)[0]
    tree_count = _int(n_trees, "n_trees", 1, _MAX_TREES)
    if tree_count * y.size * (fold_count + 1) > _MAX_WEIGHT_CELLS:
        raise ValueError("forest CV/refit exceeds the bounded total bootstrap-row budget")
    requested = np.sqrt(design.shape[1] - 1)
    mtry = min(design.shape[1], max(1, int(np.rint(requested))))
    rng = np.random.default_rng(random_state)
    scores = np.empty(fold_count, dtype=np.float64)
    fold_nodes = np.empty(fold_count, dtype=np.int64)
    fold_work = np.empty(fold_count, dtype=np.int64)
    seed_max = np.iinfo(np.int32).max
    for fold in range(fold_count):
        test = assignments == fold
        train = ~test
        seed = int(rng.integers(0, seed_max))
        forest = fit_condis_forest(
            design[train], y[train], n_trees=tree_count, mtry=mtry, nodesize=5, random_state=seed
        )
        fold_nodes[fold] = forest.node_count
        fold_work[fold] = forest.split_work
        prediction = predict_condis_forest(forest, design[test])
        scale = max(float(np.max(np.abs(y[test]))), float(np.max(np.abs(prediction))))
        if scale == 0.0:
            scale = 1.0
        normalized_rmse = float(np.sqrt(np.mean((y[test] / scale - prediction / scale) ** 2)))
        scores[fold] = scale * normalized_rmse
    if not np.isfinite(scores).all():
        raise ArithmeticError("random-forest CV produced non-finite RMSE")
    full_seed = int(rng.integers(0, seed_max))
    full_fit = fit_condis_forest(
        design, y, n_trees=tree_count, mtry=mtry, nodesize=5, random_state=full_seed
    )
    fitted = predict_condis_forest(full_fit, design)
    censored = imputation.status == 0
    below = censored & (fitted < imputation.observed_time)
    above = censored & (fitted > imputation.horizon)
    refined = np.where(censored, fitted, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    score_scale = float(np.max(np.abs(scores)))
    if score_scale == 0.0:
        mean_rmse, sd = 0.0, 0.0
    else:
        scaled_scores = scores / score_scale
        mean_rmse = score_scale * float(np.mean(scaled_scores))
        sd = score_scale * (float(np.std(scaled_scores, ddof=1)) if fold_count > 1 else 0.0)
    if not np.isfinite(mean_rmse) or not np.isfinite(sd):
        raise ArithmeticError("random-forest RMSE summaries are not representable")
    return CondiSForestRefinement(
        _owned(fitted),
        _owned(refined),
        _freeze_bool(below),
        _freeze_bool(above),
        mtry,
        _owned(scores),
        mean_rmse,
        sd,
        _freeze_int64(assignments),
        _freeze_int64(fold_nodes),
        _freeze_int64(fold_work),
        None if random_state is None else int(random_state),
        enforce_censoring,
        full_fit,
    )


def _freeze_int(values: NDArray[np.integer]) -> NDArray[np.int32]:
    result = np.array(values, dtype=np.int32, copy=True)
    result.setflags(write=False)
    return result


def _freeze_bool(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    result = np.array(values, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _freeze_int64(values: NDArray[np.integer]) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result
