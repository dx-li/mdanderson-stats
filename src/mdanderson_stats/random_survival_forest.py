"""Bounded random survival forests for ordinary right-censored outcomes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations
from math import lgamma, log

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar

_MAX_TREES = 2_000
_MAX_ROWS = 20_000
_MAX_FEATURES = 100
_MAX_DESIGN_CELLS = 2_000_000
_MAX_SAMPLE_WORK = 10_000_000
_MAX_SPLIT_WORK = 1_000_000_000
_MAX_LEAF_RECORDS = 10_000_000
_MAX_NODES = 2_000_000
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_PREDICTION_WORK = 100_000_000
_MAX_OOB_CELLS = 2_000_000
_MAX_OOB_WORK = 100_000_000
_OOB_TIE_EPSILON = 1e-9
_SPLIT_EPSILON = 1e-9
_MAX_FACTOR_SPLIT_LEVELS = 2_000_000


@dataclass(frozen=True)
class _PackedTree:
    """One binary tree and its packed leaf event-step curves."""

    feature: np.ndarray
    threshold: FloatArray
    categorical_node: np.ndarray
    split_level_offset: np.ndarray
    split_level_count: np.ndarray
    split_levels: np.ndarray
    left: np.ndarray
    right: np.ndarray
    event_offset: np.ndarray
    event_count: np.ndarray
    event_time: FloatArray
    log_survival: FloatArray
    cumulative_hazard: FloatArray


@dataclass(frozen=True)
class RandomSurvivalForestOOB:
    """Out-of-bag forest curves and source-style concordance error.

    Rows with zero tree contributors have NaN curves and mortality. The
    concordance error is ``1-C`` using the native 1e-9 absolute tie rules.
    """

    time_grid: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    contributor_count: np.ndarray
    mortality: FloatArray
    concordance_error: float
    comparable_pairs: int


@dataclass(frozen=True)
class RandomSurvivalForestFit:
    """Fitted random survival forest with numeric and optional nominal features.

    Leaf survival and Nelson--Aalen curves are stored as sparse event-time
    steps. ``covariate_mean`` is the prediction profile used when callers omit
    ``profiles``. The forest averages leaf survival and cumulative hazards
    separately, as randomForestSRC does.
    """

    time_grid: FloatArray
    covariate_mean: FloatArray
    covariate_count: int
    trees: tuple[_PackedTree, ...]
    n_trees: int
    mtry: int
    nodesize: int
    nsplit: int
    sample_fraction: float
    replace: bool
    random_state: int | None
    sampled_rows: int
    leaf_event_records: int
    node_count: int
    split_work: int
    max_depth: int
    inbag_membership: np.ndarray | None = None
    oob: RandomSurvivalForestOOB | None = None
    training_fingerprint: bytes | None = None
    categorical_levels: tuple[FloatArray | None, ...] = ()


@dataclass(frozen=True)
class RandomSurvivalForestPrediction:
    """Forest-averaged step curves, without confidence intervals."""

    profile: FloatArray
    times: FloatArray
    survival: FloatArray
    log_survival: FloatArray
    cumulative_hazard: FloatArray


@dataclass
class _Budget:
    nodes: int = 0
    split_work: int = 0
    leaf_records: int = 0
    max_depth: int = 0
    factor_split_levels: int = 0


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return result


def _budget_limit(value: object, name: str, default: int, upper: int) -> int:
    if value is None:
        return default
    return _integer(value, name, 1, upper)


def _forest_data(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if any(np.iscomplexobj(value) for value in (time, event, covariates) if value is not None):
        raise ValueError("data must be real")
    t = finite(time, "time")
    e = finite(event, "event")
    if t.ndim != 1 or not 1 <= t.size <= _MAX_ROWS or e.shape != t.shape:
        raise ValueError(f"time/event require aligned vectors with 1..{_MAX_ROWS} rows")
    if np.any(t < 0) or np.any((e != 0) & (e != 1)) or not np.any(e == 1):
        raise ValueError("time must be nonnegative; event must be binary with at least one event")
    x = np.empty((t.size, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim == 1:
        x = x[:, None]
    if (
        x.ndim != 2
        or x.shape[0] != t.size
        or x.shape[1] > _MAX_FEATURES
        or x.size > _MAX_DESIGN_CELLS
    ):
        raise ValueError(
            "covariates require one row per observation, at most 100 columns, "
            "and at most 2,000,000 design cells"
        )
    return t, e, x


def _categorical_columns(
    x: FloatArray, columns: ArrayLike | None
) -> tuple[FloatArray, tuple[FloatArray | None, ...], tuple[int, ...]]:
    if columns is None:
        return x, tuple(None for _ in range(x.shape[1])), ()
    if np.iscomplexobj(columns):
        raise ValueError("categorical_features must be real integer column indices")
    if getattr(columns, "shape", None) is None and isinstance(columns, (list, tuple)):
        if len(columns) > x.shape[1]:
            raise ValueError("categorical_features contains too many indices")
        if any(not np.isscalar(value) for value in columns):
            raise ValueError("categorical_features must be a one-dimensional index vector")
        if any(isinstance(value, (bool, np.bool_)) for value in columns):
            raise ValueError("categorical feature indices must be integers, not booleans")
    indices = np.asarray(columns)
    if indices.ndim != 1 or indices.size > x.shape[1]:
        raise ValueError("categorical_features must be a one-dimensional index vector")
    if indices.size and (
        indices.dtype.kind not in "iu"
        or indices.dtype.kind == "b"
        or np.any(indices < 0)
        or np.any(indices >= x.shape[1])
    ):
        raise ValueError("categorical feature indices must be in-range integers")
    selected = tuple(int(value) for value in indices)
    if len(set(selected)) != len(selected):
        raise ValueError("categorical_features must not contain duplicate indices")
    levels_by_column: list[FloatArray | None] = [None] * x.shape[1]
    encoded = x.copy()
    for column in selected:
        levels = np.unique(x[:, column])
        levels_by_column[column] = _freeze(levels)
        encoded[:, column] = np.searchsorted(levels, x[:, column])
    return encoded, tuple(levels_by_column), selected


def _modal_profile(
    covariates: FloatArray, categorical_levels: tuple[FloatArray | None, ...]
) -> FloatArray:
    profile = _stable_column_mean(covariates)
    for column, levels in enumerate(categorical_levels):
        if levels is not None:
            observed, counts = np.unique(covariates[:, column], return_counts=True)
            # Levels are sorted, so argmax resolves ties to the smallest label.
            profile[column] = observed[int(np.argmax(counts))]
    return profile


def _encode_profiles(
    profiles: FloatArray, categorical_levels: tuple[FloatArray | None, ...]
) -> FloatArray:
    _validate_profile_categories(profiles, categorical_levels)
    if not any(levels is not None for levels in categorical_levels):
        return profiles
    encoded = profiles.copy()
    for column, levels in enumerate(categorical_levels):
        if levels is None:
            continue
        indexes = np.searchsorted(levels, profiles[:, column])
        encoded[:, column] = indexes
    return encoded


def _validate_profile_categories(
    profiles: FloatArray, categorical_levels: tuple[FloatArray | None, ...]
) -> None:
    for column, levels in enumerate(categorical_levels):
        if levels is None:
            continue
        indexes = np.searchsorted(levels, profiles[:, column])
        valid = indexes < levels.size
        valid_indexes = np.flatnonzero(valid)
        valid[valid_indexes] = levels[indexes[valid_indexes]] == profiles[valid_indexes, column]
        if not np.all(valid):
            raise ValueError(f"profiles contain unseen level(s) for categorical feature {column}")


def _freeze_index(values: np.ndarray, dtype: np.dtype[np.signedinteger]) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _stable_column_mean(values: FloatArray) -> FloatArray:
    if values.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    magnitude = np.max(np.abs(values), axis=0)
    scaled = np.zeros(values.shape, dtype=np.float64)
    np.divide(values, magnitude, out=scaled, where=magnitude > 0)
    with np.errstate(over="ignore", invalid="ignore"):
        result = magnitude * np.mean(scaled, axis=0)
    if not np.isfinite(result).all():
        raise ArithmeticError("covariate mean exceeds numerical range")
    return result


def _time_grid(event_times: FloatArray, ntime: int | ArrayLike | None) -> FloatArray:
    if ntime is None:
        return event_times.copy()
    if np.iscomplexobj(ntime):
        raise ValueError("ntime must be real")
    raw = np.asarray(ntime)
    if raw.ndim == 1 and raw.size == 1:
        raw = raw.reshape(())
    if raw.ndim == 0:
        value = raw.item()
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
            raise ValueError("ntime must be zero or a positive integer")
        if value != int(value):
            raise ValueError("ntime must be zero or a positive integer")
        count = int(value)
        if count < 0 or count > _MAX_ROWS:
            raise ValueError(f"ntime must be in [0, {_MAX_ROWS}]")
        if count == 0 or event_times.size <= count:
            return event_times.copy()
        # R uses round(seq.int(1, n, length.out=ntime)); np.rint shares
        # its ties-to-even rounding rule. Convert 1-based indexes afterward.
        indices = np.unique(np.rint(np.linspace(1, event_times.size, count)).astype(int) - 1)
        return event_times[indices]
    requested = finite(ntime, "ntime")
    if requested.ndim != 1 or requested.size == 0 or np.any(requested < 0):
        raise ValueError("ntime must be a nonempty vector of nonnegative times")
    if requested.size > 100_000:
        raise ValueError("ntime contains too many requested times (limit 100000)")
    locations = np.searchsorted(event_times, requested, side="right") - 1
    indices = np.maximum(locations, 0)
    return np.unique(event_times[indices])


def _stop_before_split(time: FloatArray, event: FloatArray) -> bool:
    if not np.any(event == 1):
        return True
    # Native RF-SRC declines all-event parents with constant failure times.
    if np.all(event == 1) and np.all(time == time[0]):
        return True
    return False


def _tree_goes_left(tree: _PackedTree, node: int, value: float) -> bool:
    """Return the packed-tree branch for one feature value at one node.

    Kept as a single routing primitive so prediction, OOB evaluation, and
    feature-importance perturbations apply identical split semantics.
    """
    if not tree.categorical_node[node]:
        return bool(value <= tree.threshold[node])
    start = int(tree.split_level_offset[node])
    stop = start + int(tree.split_level_count[node])
    values = tree.split_levels[start:stop]
    index = int(np.searchsorted(values, int(value)))
    return bool(index < values.size and values[index] == int(value))


def _factor_split_plan(
    levels: FloatArray, node_size: int, nsplit: int
) -> tuple[int, bool, tuple[float, ...]]:
    """Choose exact versus bounded random unordered partitions for one node."""
    count = int(levels.size)
    if count < 2:
        return 0, True, ()
    partition_count = (1 << (count - 1)) - 1
    exact = count <= 32 and (
        partition_count < node_size if nsplit == 0 else partition_count <= min(node_size, nsplit)
    )
    sizes = tuple(range(1, count // 2 + 1))
    if exact:
        return partition_count, True, ()
    candidate_count = min(partition_count, node_size, nsplit if nsplit > 0 else node_size)
    log_weights = np.asarray(
        [
            lgamma(count + 1)
            - lgamma(size + 1)
            - lgamma(count - size + 1)
            - (log(2.0) if 2 * size == count else 0.0)
            for size in sizes
        ],
        dtype=np.float64,
    )
    weights = np.exp(log_weights - np.max(log_weights))
    probabilities = weights / weights.sum()
    return candidate_count, False, tuple(float(value) for value in probabilities)


def _factor_split_candidates(
    levels: FloatArray,
    candidate_count: int,
    exact: bool,
    group_probabilities: tuple[float, ...],
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray | None, float]]:
    """Yield left-level subsets without materializing a powerset."""
    count = int(levels.size)
    if exact:
        for size in range(1, count // 2 + 1):
            for subset in combinations(range(count), size):
                if 2 * size == count and 0 not in subset:
                    continue
                yield levels[np.asarray(subset, dtype=np.int64)], float("nan")
        return
    sizes = np.arange(1, count // 2 + 1, dtype=np.int64)
    for _ in range(candidate_count):
        size = int(rng.choice(sizes, p=np.asarray(group_probabilities)))
        candidate_levels = np.sort(rng.choice(levels, size=size, replace=False))
        yield candidate_levels, float("nan")


def _parent_counts(
    time: FloatArray, event: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    event_times = np.unique(time[event == 1])
    event_index = np.searchsorted(event_times, time[event == 1])
    failures = np.bincount(event_index, minlength=event_times.size).astype(np.float64)
    sorted_time = np.sort(time)
    at_risk = (time.size - np.searchsorted(sorted_time, event_times, side="left")).astype(
        np.float64
    )
    return event_times, failures, at_risk


def _logrank_score(
    time: FloatArray,
    event: FloatArray,
    left: np.ndarray,
    parent_events: FloatArray | None = None,
    parent_at_risk: FloatArray | None = None,
    event_times: FloatArray | None = None,
) -> float:
    """Native standardized two-group log-rank score for one candidate split."""
    if parent_events is None or parent_at_risk is None:
        event_times, parent_events, parent_at_risk = _parent_counts(time, event)
    if parent_events.size == 0:
        return 0.0
    if event_times is None:
        event_times = np.unique(time[event == 1])
    left_time = time[left]
    left_event = event[left]
    left_events = np.bincount(
        np.searchsorted(event_times, left_time[left_event == 1]),
        minlength=event_times.size,
    ).astype(np.float64)
    left_at_risk = (
        left_time.size - np.searchsorted(np.sort(left_time), event_times, side="left")
    ).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = left_at_risk * parent_events / parent_at_risk
        numerator = float(np.sum(left_events - expected))
        fraction = left_at_risk / parent_at_risk
        finite_population = (parent_at_risk - parent_events) / (parent_at_risk - 1)
        variance_terms = np.where(
            parent_at_risk >= 2,
            fraction * (1 - fraction) * finite_population * parent_events,
            0.0,
        )
    denominator = float(np.sqrt(np.sum(variance_terms)))
    numerator = abs(numerator)
    if denominator <= 1e-9:
        return 0.0 if numerator <= 1e-9 else float("inf")
    return numerator / denominator


def _leaf_curve(time: FloatArray, event: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    event_times, failures, at_risk = _parent_counts(time, event)
    if event_times.size == 0:
        return event_times, event_times.copy(), event_times.copy()
    ratios = failures / at_risk
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.cumsum(np.log1p(-ratios))
    cumulative_hazard = np.cumsum(ratios)
    return event_times, log_survival, cumulative_hazard


def _pack_tree(
    feature: list[int],
    threshold: list[float],
    categorical_node: list[bool],
    split_level_offset: list[int],
    split_level_count: list[int],
    split_levels: list[int],
    left: list[int],
    right: list[int],
    event_offset: list[int],
    event_count: list[int],
    event_time: list[FloatArray],
    log_survival: list[FloatArray],
    cumulative_hazard: list[FloatArray],
) -> _PackedTree:
    return _PackedTree(
        _freeze_index(np.asarray(feature), np.dtype(np.int32)),
        _freeze(np.asarray(threshold, dtype=np.float64)),
        np.frombuffer(np.asarray(categorical_node, dtype=np.bool_).tobytes(), dtype=np.bool_),
        _freeze_index(np.asarray(split_level_offset), np.dtype(np.int32)),
        _freeze_index(np.asarray(split_level_count), np.dtype(np.int32)),
        _freeze_index(np.asarray(split_levels), np.dtype(np.int32)),
        _freeze_index(np.asarray(left), np.dtype(np.int32)),
        _freeze_index(np.asarray(right), np.dtype(np.int32)),
        _freeze_index(np.asarray(event_offset), np.dtype(np.int64)),
        _freeze_index(np.asarray(event_count), np.dtype(np.int32)),
        _freeze(np.concatenate(event_time) if event_time else np.empty(0)),
        _freeze(np.concatenate(log_survival) if log_survival else np.empty(0)),
        _freeze(np.concatenate(cumulative_hazard) if cumulative_hazard else np.empty(0)),
    )


def _grow_tree(
    x: FloatArray,
    time: FloatArray,
    event: FloatArray,
    bootstrap_rows: np.ndarray,
    *,
    rng: np.random.Generator,
    mtry: int,
    nodesize: int,
    nsplit: int,
    budget: _Budget,
    max_nodes: int,
    max_split_work: int,
    max_leaf_records: int,
    categorical_columns: frozenset[int],
) -> _PackedTree:
    feature = [-1]
    threshold = [np.nan]
    categorical_node = [False]
    split_level_offset = [0]
    split_level_count = [0]
    split_levels: list[int] = []
    left_child = [-1]
    right_child = [-1]
    event_offset = [0]
    event_count = [0]
    step_times: list[FloatArray] = []
    step_log_survival: list[FloatArray] = []
    step_hazard: list[FloatArray] = []
    event_cursor = 0
    budget.nodes += 1
    if budget.nodes > max_nodes:
        raise ValueError("forest exceeds max_nodes budget")
    feature_count = x.shape[1]
    initial_features = np.ones(feature_count, dtype=bool)
    stack: list[tuple[int, np.ndarray, np.ndarray, int]] = [
        (0, bootstrap_rows, initial_features, 0)
    ]

    while stack:
        node, rows, permissible, depth = stack.pop()
        budget.max_depth = max(budget.max_depth, depth)
        node_time = time[rows]
        node_event = event[rows]
        best_feature = -1
        best_threshold = np.nan
        best_levels: np.ndarray | None = None
        best_score = -np.inf
        next_permissible = permissible.copy()
        if rows.size >= 2 * nodesize and not _stop_before_split(node_time, node_event):
            candidate_features = np.flatnonzero(permissible)
            if candidate_features.size:
                selected = rng.choice(
                    candidate_features,
                    size=min(mtry, candidate_features.size),
                    replace=False,
                )
                parent_event_times, parent_events, parent_at_risk = _parent_counts(
                    node_time, node_event
                )
                for column in selected:
                    unique_values = np.unique(x[rows, column])
                    if unique_values.size < 2:
                        next_permissible[column] = False
                        continue
                    candidate_iterator: Iterator[tuple[np.ndarray | None, float]]
                    if int(column) in categorical_columns:
                        candidate_count, exact, group_probabilities = _factor_split_plan(
                            unique_values, int(rows.size), nsplit
                        )
                        budget.split_work += int(candidate_count * (rows.size + unique_values.size))
                        candidate_iterator = _factor_split_candidates(
                            unique_values,
                            candidate_count,
                            exact,
                            group_probabilities,
                            rng,
                        )
                    else:
                        cuts = unique_values[:-1]
                        if nsplit > 0 and cuts.size > nsplit:
                            cuts = np.sort(rng.choice(cuts, size=nsplit, replace=False))
                        candidate_count = int(cuts.size)
                        budget.split_work += int(rows.size * cuts.size)
                        candidate_iterator = ((None, float(cut)) for cut in cuts)
                    if budget.split_work > max_split_work:
                        raise ValueError("forest exceeds max_split_work budget")
                    for left_levels, cut in candidate_iterator:
                        left_mask = (
                            np.isin(x[rows, column], left_levels)
                            if left_levels is not None
                            else x[rows, column] <= cut
                        )
                        left_count = int(np.count_nonzero(left_mask))
                        if left_count == 0 or left_count == rows.size:
                            continue
                        score = _logrank_score(
                            node_time,
                            node_event,
                            left_mask,
                            parent_events,
                            parent_at_risk,
                            parent_event_times,
                        )
                        if score - best_score > _SPLIT_EPSILON:
                            best_score = score
                            best_feature = int(column)
                            best_threshold = float(cut)
                            best_levels = left_levels

        if best_feature < 0:
            times, log_curve, hazard_curve = _leaf_curve(node_time, node_event)
            budget.leaf_records += int(times.size)
            if budget.leaf_records > max_leaf_records:
                raise ValueError("forest exceeds max_leaf_event_records budget")
            event_offset[node] = event_cursor
            event_count[node] = int(times.size)
            if times.size:
                step_times.append(times)
                step_log_survival.append(log_curve)
                step_hazard.append(hazard_curve)
                event_cursor += int(times.size)
            continue

        if best_levels is not None:
            budget.factor_split_levels += int(best_levels.size)
            if budget.factor_split_levels > _MAX_FACTOR_SPLIT_LEVELS:
                raise ValueError("forest exceeds packed categorical split-level budget")
            categorical_node[node] = True
            split_level_offset[node] = len(split_levels)
            split_level_count[node] = int(best_levels.size)
            split_levels.extend(int(level) for level in best_levels)
            left_mask = np.isin(x[rows, best_feature], best_levels)
        else:
            left_mask = x[rows, best_feature] <= best_threshold
        left_rows = rows[left_mask]
        right_rows = rows[~left_mask]
        left_index = len(feature)
        right_index = left_index + 1
        feature[node] = best_feature
        threshold[node] = best_threshold
        left_child[node] = left_index
        right_child[node] = right_index
        for _ in range(2):
            budget.nodes += 1
            if budget.nodes > max_nodes:
                raise ValueError("forest exceeds max_nodes budget")
            feature.append(-1)
            threshold.append(np.nan)
            categorical_node.append(False)
            split_level_offset.append(0)
            split_level_count.append(0)
            left_child.append(-1)
            right_child.append(-1)
            event_offset.append(0)
            event_count.append(0)
        # Native trees process the left branch first; preserve that random
        # draw order by pushing right before left on this LIFO work list.
        stack.append((right_index, right_rows, next_permissible, depth + 1))
        stack.append((left_index, left_rows, next_permissible, depth + 1))

    return _pack_tree(
        feature,
        threshold,
        categorical_node,
        split_level_offset,
        split_level_count,
        split_levels,
        left_child,
        right_child,
        event_offset,
        event_count,
        step_times,
        step_log_survival,
        step_hazard,
    )


def _forest_fingerprint(
    time: FloatArray,
    event: FloatArray,
    x: FloatArray,
    categorical_levels: tuple[FloatArray | None, ...] = (),
) -> bytes:
    """Hash normalized training values and row order without a joined copy."""
    digest = blake2b(digest_size=20)
    for values in (time, event, x):
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        if contiguous.nbytes:
            digest.update(memoryview(contiguous).cast("B"))
    for levels in categorical_levels:
        if levels is None:
            digest.update(b"N")
        else:
            contiguous = np.ascontiguousarray(levels, dtype=np.float64)
            digest.update(b"C")
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(memoryview(contiguous).cast("B"))
    return digest.digest()


def _oob_concordance_error(
    time: FloatArray, event: FloatArray, mortality: FloatArray, contributors: np.ndarray
) -> tuple[float, int]:
    """Native getConcordanceIndex pair rules, without an n-by-n matrix."""
    eligible = contributors > 0
    pairs = concordant = 0
    n = time.size
    for i in range(n - 1):
        if not eligible[i]:
            continue
        j = np.arange(i + 1, n, dtype=np.int64)
        use = eligible[j]
        j = j[use]
        if j.size == 0:
            continue
        delta_time = time[i] - time[j]
        tied_time = np.abs(delta_time) <= _OOB_TIE_EPSILON
        both_event = tied_time & (event[i] == 1) & (event[j] == 1)
        event_i_first = ((delta_time < -_OOB_TIE_EPSILON) & (event[i] == 1)) | (
            tied_time & (event[i] == 1) & (event[j] == 0)
        )
        event_j_first = ((delta_time > _OOB_TIE_EPSILON) & (event[j] == 1)) | (
            tied_time & (event[j] == 1) & (event[i] == 0)
        )
        comparable = both_event | event_i_first | event_j_first
        if not np.any(comparable):
            continue
        j = j[comparable]
        both_event = both_event[comparable]
        event_i_first = event_i_first[comparable]
        delta_risk = np.where(
            event_i_first, mortality[i] - mortality[j], mortality[j] - mortality[i]
        )
        absolute_delta = np.abs(delta_risk)
        pair_concordance = np.where(
            both_event,
            np.where(absolute_delta < _OOB_TIE_EPSILON, 2, 1),
            np.where(
                delta_risk > _OOB_TIE_EPSILON,
                2,
                np.where(absolute_delta <= _OOB_TIE_EPSILON, 1, 0),
            ),
        )
        pairs += int(j.size)
        concordant += int(pair_concordance.sum())
    if pairs == 0:
        return float("nan"), 0
    return 1.0 - concordant / (2.0 * pairs), pairs


def _oob_curves(
    time: FloatArray,
    event: FloatArray,
    covariates: FloatArray,
    trees: tuple[_PackedTree, ...],
    time_grid: FloatArray,
    membership: np.ndarray,
    *,
    max_depth: int,
    max_work: int,
) -> RandomSurvivalForestOOB:
    n = time.size
    oob_rows = 0
    for packed in membership:
        inbag_count = int(np.unpackbits(packed, bitorder="little")[:n].sum())
        oob_rows += n - inbag_count
    work = oob_rows * (time_grid.size + max_depth) + n * n
    if work > max_work:
        raise ValueError("OOB curve and concordance work exceeds max_oob_work budget")
    survival_sum = np.zeros((n, time_grid.size), dtype=np.float64)
    hazard_sum = np.zeros_like(survival_sum)
    contributors = np.zeros(n, dtype=np.int64)
    for tree_index, tree in enumerate(trees):
        inbag = np.unpackbits(membership[tree_index], bitorder="little")[:n].astype(bool)
        for row in np.flatnonzero(~inbag):
            node = 0
            profile = covariates[row]
            while tree.feature[node] >= 0:
                column = int(tree.feature[node])
                node = (
                    int(tree.left[node])
                    if _tree_goes_left(tree, node, float(profile[column]))
                    else int(tree.right[node])
                )
            offset = int(tree.event_offset[node])
            count = int(tree.event_count[node])
            if count:
                steps = slice(offset, offset + count)
                index = np.searchsorted(tree.event_time[steps], time_grid, side="right") - 1
                included = index >= 0
                leaf_survival = np.ones(time_grid.size, dtype=np.float64)
                leaf_hazard = np.zeros(time_grid.size, dtype=np.float64)
                if np.any(included):
                    leaf_survival[included] = np.exp(tree.log_survival[offset + index[included]])
                    leaf_hazard[included] = tree.cumulative_hazard[offset + index[included]]
                survival_sum[row] += leaf_survival
                hazard_sum[row] += leaf_hazard
            else:
                survival_sum[row] += 1.0
            contributors[row] += 1
    survival = np.full_like(survival_sum, np.nan)
    cumulative_hazard = np.full_like(hazard_sum, np.nan)
    present = contributors > 0
    survival[present] = survival_sum[present] / contributors[present, None]
    cumulative_hazard[present] = hazard_sum[present] / contributors[present, None]
    mortality = np.full(n, np.nan, dtype=np.float64)
    mortality[present] = cumulative_hazard[present].sum(axis=1)
    concordance_error, comparable_pairs = _oob_concordance_error(
        time, event, mortality, contributors
    )
    return RandomSurvivalForestOOB(
        _freeze(time_grid),
        _freeze(survival),
        _freeze(cumulative_hazard),
        np.frombuffer(contributors.tobytes(), dtype=np.int64),
        _freeze(mortality),
        concordance_error,
        comparable_pairs,
    )


def fit_random_survival_forest(
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    categorical_features: ArrayLike | None = None,
    n_trees: int = 500,
    mtry: int | None = None,
    nodesize: int = 15,
    nsplit: int = 10,
    replace: bool = False,
    sample_fraction: float | None = None,
    ntime: int | ArrayLike | None = 150,
    random_state: int | None = None,
    max_sampled_rows: int = 2_000_000,
    max_split_work: int = 100_000_000,
    max_leaf_event_records: int = 2_000_000,
    max_nodes: int = 1_000_000,
    compute_oob: bool = False,
    max_oob_cells: int = _MAX_OOB_CELLS,
    max_oob_work: int = _MAX_OOB_WORK,
) -> RandomSurvivalForestFit:
    """Fit log-rank trees for right-censored data with numeric/nominal features.

    Events use status 1 and right censoring uses status 0. Trees use axis-aligned
    Numeric features use ordered cuts; ``categorical_features`` names nominal
    columns, whose values are finite numeric labels and whose nodes split by
    unordered subsets. Training levels are mapped once and retained; unknown
    prediction levels are rejected. Exact subset enumeration is used only for
    small source-compatible candidate sets, otherwise subsets are sampled
    without materializing a powerset. Kaplan--Meier terminal
    survival curves. With replacement disabled, the default sample fraction is
    0.632; with replacement enabled it is 1.0. These defaults follow
    randomForestSRC 3.2.2, but NumPy's random stream does not match R's.
    Set ``compute_oob=True`` to retain bit-packed in-bag membership and compute
    OOB curves, mortality and concordance error; the default avoids this storage
    and work. Rows without an OOB tree remain undefined (NaN), never in-bag-filled.
    """
    if not isinstance(replace, (bool, np.bool_)):
        raise ValueError("replace must be boolean")
    if not isinstance(compute_oob, (bool, np.bool_)):
        raise ValueError("compute_oob must be boolean")
    oob_cell_limit = _budget_limit(max_oob_cells, "max_oob_cells", _MAX_OOB_CELLS, _MAX_OOB_CELLS)
    oob_work_limit = _budget_limit(max_oob_work, "max_oob_work", _MAX_OOB_WORK, _MAX_OOB_WORK)
    tree_count = _integer(n_trees, "n_trees", 1, _MAX_TREES)
    leaf_size = _integer(nodesize, "nodesize", 1, _MAX_ROWS)
    random_splits = _integer(nsplit, "nsplit", 0, _MAX_ROWS)
    t, e, raw_x = _forest_data(time, event, covariates)
    x, categorical_levels, categorical = _categorical_columns(raw_x, categorical_features)
    categorical_set = frozenset(categorical)
    if x.shape[1] > _MAX_FEATURES:
        raise ValueError(f"covariates may have at most {_MAX_FEATURES} columns")
    if x.shape[1] == 0:
        if mtry is not None:
            raise ValueError("mtry must be omitted when no covariates are supplied")
        feature_count = 0
    elif mtry is None:
        feature_count = int(np.ceil(np.sqrt(x.shape[1])))
    else:
        feature_count = _integer(mtry, "mtry", 1, x.shape[1])
    fraction_default = 1.0 if replace else 0.632
    if sample_fraction is None:
        fraction = fraction_default
    else:
        if np.iscomplexobj(sample_fraction):
            raise ValueError("sample_fraction must be real")
        if isinstance(sample_fraction, (bool, np.bool_)):
            raise ValueError("sample_fraction must be a finite number in (0, 1]")
        fraction = scalar(sample_fraction, "sample_fraction")
        if not 0 < fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
    seed: int | None
    if random_state is None:
        seed = None
    else:
        seed = _integer(random_state, "random_state", 0, np.iinfo(np.int32).max)
    sample_limit = _budget_limit(max_sampled_rows, "max_sampled_rows", 2_000_000, _MAX_SAMPLE_WORK)
    split_limit = _budget_limit(max_split_work, "max_split_work", 100_000_000, _MAX_SPLIT_WORK)
    leaf_limit = _budget_limit(
        max_leaf_event_records,
        "max_leaf_event_records",
        2_000_000,
        _MAX_LEAF_RECORDS,
    )
    node_limit = _budget_limit(max_nodes, "max_nodes", 1_000_000, _MAX_NODES)
    sample_size = int(np.rint(fraction * t.size))
    if not replace:
        sample_size = min(sample_size, int(t.size))
    if sample_size < 1:
        raise ValueError("sample_fraction rounds to an empty tree sample")
    sampled_rows = tree_count * sample_size
    if sampled_rows > sample_limit:
        raise ValueError("forest exceeds max_sampled_rows budget")
    event_times = np.unique(t[e == 1])
    output_grid = _time_grid(event_times, ntime)
    membership: np.ndarray | None = None
    if compute_oob:
        packed_width = (t.size + 7) // 8
        membership_cells = tree_count * packed_width
        output_cells = t.size * output_grid.size
        combined_cells = 2 * membership_cells + 8 * output_cells + 16 * t.size
        if combined_cells > oob_cell_limit:
            raise ValueError("OOB membership and curve outputs exceed max_oob_cells budget")
        if t.size * t.size > oob_work_limit:
            raise ValueError("OOB pairwise-concordance work exceeds max_oob_work budget")
        membership = np.zeros((tree_count, packed_width), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    budget = _Budget()
    trees: list[_PackedTree] = []
    for tree_index in range(tree_count):
        bootstrap = rng.choice(t.size, size=sample_size, replace=bool(replace))
        if membership is not None:
            inbag = np.zeros(t.size, dtype=np.uint8)
            inbag[bootstrap] = 1
            membership[tree_index] = np.packbits(inbag, bitorder="little")
        trees.append(
            _grow_tree(
                x,
                t,
                e,
                bootstrap,
                rng=rng,
                mtry=feature_count,
                nodesize=leaf_size,
                nsplit=random_splits,
                budget=budget,
                max_nodes=node_limit,
                max_split_work=split_limit,
                max_leaf_records=leaf_limit,
                categorical_columns=categorical_set,
            )
        )
    packed_membership = (
        None
        if membership is None
        else np.frombuffer(membership.tobytes(), dtype=np.uint8).reshape(membership.shape)
    )
    oob = (
        None
        if membership is None
        else _oob_curves(
            t,
            e,
            x,
            tuple(trees),
            output_grid,
            membership,
            max_depth=budget.max_depth,
            max_work=oob_work_limit,
        )
    )
    return RandomSurvivalForestFit(
        _freeze(output_grid),
        _freeze(_modal_profile(raw_x, categorical_levels)),
        x.shape[1],
        tuple(trees),
        tree_count,
        feature_count,
        leaf_size,
        random_splits,
        fraction,
        bool(replace),
        seed,
        sampled_rows,
        budget.leaf_records,
        budget.nodes,
        budget.split_work,
        budget.max_depth,
        packed_membership,
        oob,
        _forest_fingerprint(t, e, raw_x, categorical_levels if categorical else ())
        if compute_oob
        else None,
        categorical_levels,
    )


def predict_random_survival_forest(
    fit: RandomSurvivalForestFit,
    times: ArrayLike | None = None,
    profiles: ArrayLike | None = None,
    *,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
    max_prediction_work: int = _MAX_PREDICTION_WORK,
) -> RandomSurvivalForestPrediction:
    """Predict the separately averaged survival and cumulative-hazard curves."""
    if times is None:
        time_values = np.asarray(fit.time_grid)
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        time_values = finite(times, "times")
    if (
        time_values.ndim != 1
        or time_values.size == 0
        or time_values.size > 100_000
        or np.any(time_values < 0)
    ):
        raise ValueError("times must be a vector of 1..100,000 nonnegative values")
    if profiles is None:
        profile_values = np.asarray(fit.covariate_mean)[None, :]
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile_values = finite(profiles, "profiles")
        if profile_values.ndim == 1:
            profile_values = profile_values[None, :]
    if (
        profile_values.ndim != 2
        or profile_values.shape[1] != fit.covariate_count
        or profile_values.shape[0] == 0
        or profile_values.shape[0] > 100_000
        or profile_values.size > _MAX_DESIGN_CELLS
    ):
        raise ValueError("profiles must have one column per fitted covariate")
    categorical_levels = fit.categorical_levels or tuple(None for _ in range(fit.covariate_count))
    output_limit = _integer(max_output_cells, "max_output_cells", 1, _MAX_OUTPUT_CELLS)
    work_limit = _integer(max_prediction_work, "max_prediction_work", 1, _MAX_PREDICTION_WORK)
    cells = int(profile_values.shape[0] * time_values.size)
    categorical_copy = (
        profile_values.size if any(levels is not None for levels in categorical_levels) else 0
    )
    combined_cells = 8 * cells + profile_values.size + time_values.size + categorical_copy
    if combined_cells > output_limit:
        raise ValueError("prediction exceeds max_output_cells budget")
    work = cells * fit.n_trees + profile_values.shape[0] * fit.n_trees * fit.max_depth
    if work > work_limit:
        raise ValueError("prediction exceeds max_prediction_work budget")
    routed_profiles = _encode_profiles(profile_values, categorical_levels)
    log_survival_sum = np.full((profile_values.shape[0], time_values.size), -np.inf)
    hazard_sum = np.zeros_like(log_survival_sum)
    for tree in fit.trees:
        for profile_index, profile in enumerate(routed_profiles):
            node = 0
            while tree.feature[node] >= 0:
                column = int(tree.feature[node])
                node = (
                    int(tree.left[node])
                    if _tree_goes_left(tree, node, float(profile[column]))
                    else int(tree.right[node])
                )
            offset = int(tree.event_offset[node])
            count = int(tree.event_count[node])
            if count:
                steps = slice(offset, offset + count)
                index = np.searchsorted(tree.event_time[steps], time_values, side="right") - 1
                has_step = index >= 0
                leaf_log_survival = np.zeros(time_values.size)
                leaf_hazard = np.zeros(time_values.size)
                if np.any(has_step):
                    leaf_log_survival[has_step] = tree.log_survival[offset + index[has_step]]
                    leaf_hazard[has_step] = tree.cumulative_hazard[offset + index[has_step]]
                log_survival_sum[profile_index] = np.logaddexp(
                    log_survival_sum[profile_index], leaf_log_survival
                )
                hazard_sum[profile_index] += leaf_hazard
            else:
                log_survival_sum[profile_index] = np.logaddexp(log_survival_sum[profile_index], 0.0)
    log_survival = np.minimum(log_survival_sum - log(fit.n_trees), 0.0)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        survival = np.exp(log_survival)
    cumulative_hazard = hazard_sum / fit.n_trees
    return RandomSurvivalForestPrediction(
        _freeze(profile_values),
        _freeze(time_values),
        _freeze(survival),
        _freeze(log_survival),
        _freeze(cumulative_hazard),
    )
