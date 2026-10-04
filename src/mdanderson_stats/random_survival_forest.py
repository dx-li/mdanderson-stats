"""Bounded random survival forests for ordinary right-censored outcomes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations
from math import lgamma, log
from typing import Literal

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
_MAX_IMPUTATION_CELLS = 2_000_000
_OOB_TIE_EPSILON = 1e-9
_SPLIT_EPSILON = 1e-9
_MAX_FACTOR_SPLIT_LEVELS = 2_000_000
_SPLIT_RULES = ("logrank", "logrankscore", "bs.gradient", "random")
_DEFAULT_BRIER_PROB = 0.90


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
    represented_count: np.ndarray | None = None
    imputation_donors: tuple[tuple[FloatArray, ...], ...] | None = None
    training_leaf_node: np.ndarray | None = None


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
    row_indices: np.ndarray | None = None
    concordance_available: bool = True


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
    split_rule: Literal["logrank", "logrankscore", "bs.gradient", "random"] = "logrank"
    split_probability: float | None = None
    na_action: Literal["raise", "omit", "impute"] = "raise"
    imputation_performed: bool = False
    training_row_indices: np.ndarray | None = None
    original_training_fingerprint: bytes | None = None
    requested_trees: int | None = None


@dataclass(frozen=True)
class RandomSurvivalForestPrediction:
    """Forest-averaged step curves, without confidence intervals."""

    profile: FloatArray
    times: FloatArray
    survival: FloatArray
    log_survival: FloatArray
    cumulative_hazard: FloatArray
    row_indices: np.ndarray | None = None


@dataclass
class _Budget:
    nodes: int = 0
    split_work: int = 0
    leaf_records: int = 0
    max_depth: int = 0
    factor_split_levels: int = 0
    imputation_cells: int = 0


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


def _forest_data_allow_missing(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Normalize bounded real forest inputs while retaining NaN missingness."""
    _preflight_forest_vector(time, "time")
    _preflight_forest_vector(event, "event")
    if covariates is not None:
        _preflight_forest_matrix(covariates)
    if any(np.iscomplexobj(value) for value in (time, event, covariates) if value is not None):
        raise ValueError("data must be real")
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=np.float64)
    if t.ndim != 1 or not 1 <= t.size <= _MAX_ROWS or e.shape != t.shape:
        raise ValueError(f"time/event require aligned vectors with 1..{_MAX_ROWS} rows")
    x = (
        np.empty((t.size, 0), dtype=np.float64)
        if covariates is None
        else np.asarray(covariates, dtype=np.float64)
    )
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
    if (
        np.any(np.isinf(t))
        or np.any(np.isinf(e))
        or np.any(np.isinf(x))
        or np.any(t[np.isfinite(t)] < 0)
        or np.any(~np.isin(e[np.isfinite(e)], (0.0, 1.0)))
    ):
        raise ValueError("observed times must be nonnegative and observed status must be binary")
    return t, e, x


def _preflight_forest_vector(value: object, name: str) -> None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 1 or shape[0] > _MAX_ROWS:
            raise ValueError(f"{name} must be a vector with at most {_MAX_ROWS} rows")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ROWS or any(
            not np.isscalar(item) or isinstance(item, (list, tuple)) for item in value
        ):
            raise ValueError(f"{name} must be a bounded one-dimensional vector")
        return
    raise ValueError(f"{name} must be a bounded array or list/tuple vector")


def _preflight_forest_matrix(value: object) -> None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) not in (1, 2):
            raise ValueError("covariates must be a bounded vector or matrix")
        if shape[0] > _MAX_ROWS:
            raise ValueError(f"covariates may have at most {_MAX_ROWS} rows")
        if len(shape) == 2 and (
            shape[1] > _MAX_FEATURES or shape[0] * shape[1] > _MAX_DESIGN_CELLS
        ):
            raise ValueError("covariates exceed the feature or design-cell limit")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ROWS:
            raise ValueError(f"covariates may have at most {_MAX_ROWS} rows")
        if value and all(np.isscalar(item) for item in value):
            if len(value) > _MAX_DESIGN_CELLS:
                raise ValueError("covariates exceed the design-cell limit")
            return
        if len(value) == 0:
            return
        first = value[0]
        if not isinstance(first, (list, tuple, np.ndarray)) or len(first) > _MAX_FEATURES:
            raise ValueError("covariates must be a bounded rectangular matrix")
        columns = len(first)
        if len(value) * columns > _MAX_DESIGN_CELLS:
            raise ValueError("covariates exceed the design-cell limit")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != columns:
                raise ValueError("covariates must be a bounded rectangular matrix")
            if any(not np.isscalar(item) for item in row):
                raise ValueError("covariates must be a two-dimensional numeric matrix")
        return
    raise ValueError("covariates must be a bounded array or list/tuple matrix")


def _preflight_profile_matrix(value: object) -> None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) not in (1, 2):
            raise ValueError("profiles must be a bounded vector or matrix")
        rows = 1 if len(shape) == 1 else shape[0]
        columns = shape[0] if len(shape) == 1 else shape[1]
        if rows > 100_000 or columns > _MAX_FEATURES or rows * columns > _MAX_DESIGN_CELLS:
            raise ValueError("profiles exceed the prediction row, feature, or cell limit")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 100_000:
            raise ValueError("profiles exceed the prediction row limit")
        if value and all(np.isscalar(item) for item in value):
            if len(value) > _MAX_FEATURES:
                raise ValueError("profile has too many features")
            return
        if value and not isinstance(value[0], (list, tuple, np.ndarray)):
            raise ValueError("profiles must be a bounded rectangular matrix")
        columns = len(value[0]) if value else 0
        if columns > _MAX_FEATURES or len(value) * columns > _MAX_DESIGN_CELLS:
            raise ValueError("profiles exceed the prediction feature or cell limit")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != columns:
                raise ValueError("profiles must be a bounded rectangular matrix")
            if any(not np.isscalar(item) for item in row):
                raise ValueError("profiles must be a two-dimensional numeric matrix")
        return
    raise ValueError("profiles must be a bounded array or list/tuple matrix")


def _forest_original_fingerprint(
    time: FloatArray,
    event: FloatArray,
    x: FloatArray,
    categorical_features: tuple[int, ...] = (),
) -> bytes:
    """Hash raw-order values including NaN masks for adapter input checks."""
    digest = blake2b(digest_size=20)
    for values in (time, event, x):
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        if contiguous.nbytes:
            digest.update(memoryview(contiguous).cast("B"))
    digest.update(b"categorical")
    digest.update(np.asarray(categorical_features, dtype=np.int64).tobytes())
    return digest.digest()


def _impute_node_values(
    values: FloatArray,
    missing_mask: np.ndarray,
    donor_rows: np.ndarray,
    recipient_rows: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[FloatArray, np.ndarray]:
    """Draw node-local replacements, preserving bootstrap donor multiplicity.

    Empty child pools retain values imputed at an ancestor. Donor row indices
    are returned so the route can be audited without retaining full matrices.
    """
    recipients = np.asarray(recipient_rows, dtype=np.int64)
    donor_indices = np.asarray(donor_rows, dtype=np.int64)
    output = np.asarray(values[recipients], dtype=np.float64).copy()
    selected_donors = np.full(recipients.size, -1, dtype=np.int64)
    missing = np.asarray(missing_mask[recipients], dtype=bool)
    eligible = donor_indices[~missing_mask[donor_indices]]
    positions = np.flatnonzero(missing)
    if positions.size and eligible.size:
        draws = rng.integers(eligible.size, size=positions.size)
        selected_donors[positions] = eligible[draws]
        output[positions] = values[selected_donors[positions]]
    return output, selected_donors


def _terminal_impute_value(
    values: FloatArray,
    missing_mask: np.ndarray,
    donor_rows: np.ndarray,
    fallback_rows: np.ndarray,
    *,
    kind: Literal["time", "categorical"],
    master_times: FloatArray,
    rng: np.random.Generator,
) -> float | None:
    """Summarize terminal donors; fall back to ancestor-imputed in-bag rows."""
    donor_indices = np.asarray(donor_rows, dtype=np.int64)
    observed = donor_indices[~missing_mask[donor_indices]]
    source_rows = observed
    if source_rows.size == 0:
        source_rows = np.asarray(fallback_rows, dtype=np.int64)
    if source_rows.size == 0:
        return None
    sample = values[source_rows]
    if kind == "time":
        estimate = float(_stable_column_mean(sample.reshape(-1, 1))[0])
        if not np.isfinite(estimate):
            raise ArithmeticError("terminal time imputation mean is not finite")
        if estimate < master_times[0] - 1e-9 or estimate > master_times[-1] + 1e-9:
            raise ArithmeticError("terminal time mean falls outside the master-time range")
        index = int(np.searchsorted(master_times, estimate, side="left"))
        if index == 0:
            return float(master_times[0])
        if index == master_times.size:
            return float(master_times[-1])
        lower = float(master_times[index - 1])
        upper = float(master_times[index])
        left_distance = estimate - lower
        right_distance = upper - estimate
        if left_distance < right_distance:
            return lower
        if abs(left_distance - right_distance) < 1e-9:
            return lower if rng.random() <= 0.5 else upper
        return upper
    unique, counts = np.unique(sample, return_counts=True)
    modes = unique[counts == counts.max()]
    return float(modes[0] if modes.size == 1 else modes[int(rng.integers(modes.size))])


def _categorical_columns(
    x: FloatArray,
    columns: ArrayLike | None,
    observed_mask: np.ndarray | None = None,
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
        observed = (
            np.ones(x.shape[0], dtype=bool) if observed_mask is None else ~observed_mask[:, column]
        )
        levels = np.unique(x[observed, column])
        if not levels.size:
            raise ValueError("categorical predictor has no observed donor levels")
        levels_by_column[column] = _freeze(levels)
        encoded[observed, column] = np.searchsorted(levels, x[observed, column])
        encoded[~observed, column] = 0.0
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


def _modal_profile_observed(
    covariates: FloatArray,
    missing_mask: np.ndarray,
    categorical_levels: tuple[FloatArray | None, ...],
) -> FloatArray:
    profile = np.zeros(covariates.shape[1], dtype=np.float64)
    for column, levels in enumerate(categorical_levels):
        observed = ~missing_mask[:, column]
        values = covariates[observed, column]
        if not values.size:
            raise ValueError(f"predictor {column} has no observed values")
        if levels is None:
            profile[column] = float(_stable_column_mean(values.reshape(-1, 1))[0])
        else:
            unique, counts = np.unique(values, return_counts=True)
            profile[column] = float(unique[int(np.argmax(counts))])
    return profile


def _encode_profiles(
    profiles: FloatArray,
    categorical_levels: tuple[FloatArray | None, ...],
    missing_mask: np.ndarray | None = None,
) -> FloatArray:
    if missing_mask is None or not np.any(missing_mask):
        _validate_profile_categories(profiles, categorical_levels)
    else:
        for column, levels in enumerate(categorical_levels):
            if levels is None:
                continue
            observed = ~missing_mask[:, column]
            _validate_profile_categories(
                profiles[observed],
                tuple(levels if index == column else None for index in range(profiles.shape[1])),
            )
    if not any(levels is not None for levels in categorical_levels):
        return profiles
    encoded = profiles.copy()
    for column, levels in enumerate(categorical_levels):
        if levels is None:
            continue
        indexes = np.searchsorted(levels, profiles[:, column])
        observed = (
            np.ones(profiles.shape[0], dtype=bool)
            if missing_mask is None
            else ~missing_mask[:, column]
        )
        encoded[observed, column] = indexes[observed]
        encoded[~observed, column] = 0.0
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


def _random_factor_group_probabilities(count: int) -> tuple[np.ndarray, FloatArray]:
    """Return RF-SRC partition cardinalities and their native group masses."""
    group_sizes = np.arange(1, count // 2 + 1, dtype=np.int64)
    if count < 2:
        raise ValueError("a random factor split requires at least two levels")
    log_weights = np.asarray(
        [
            lgamma(count + 1)
            - lgamma(int(size) + 1)
            - lgamma(count - int(size) + 1)
            - (log(2.0) if 2 * int(size) == count else 0.0)
            for size in group_sizes
        ],
        dtype=np.float64,
    )
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    return group_sizes, probabilities


def _random_factor_partition(levels: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Draw one RF-SRC unordered factor partition with uniform partition mass.

    RF-SRC first weights each represented partition-cardinality group by its
    number of complementary partitions, then samples a subset uniformly
    within that group. A balanced split is counted once after complementing.
    """
    group_sizes, probabilities = _random_factor_group_probabilities(int(levels.size))
    group_size = int(rng.choice(group_sizes, p=probabilities))
    return np.sort(rng.choice(levels, size=group_size, replace=False))


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


def _logrankscore_scores(time: FloatArray, event: FloatArray) -> FloatArray:
    """Hothorn--Lausen individual scores with maximum-rank time ties.

    This is the published/source-vignette convention: every equal observed
    time receives its maximum rank, and each event-time group contributes
    ``d / (n - rank + 1)`` to the cumulative score term.
    """
    n = int(time.size)
    event_times, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    events = np.bincount(inverse, weights=event, minlength=event_times.size)
    rank = np.cumsum(counts, dtype=np.int64)
    increments = events / (n - rank + 1.0)
    cumulative = np.cumsum(increments)
    return event - cumulative[inverse]


def _logrankscore_split_score(scores: FloatArray, left: np.ndarray) -> float:
    """Absolute standardized Hothorn--Lausen score for a daughter split."""
    n = int(scores.size)
    n_left = int(np.count_nonzero(left))
    if n_left == 0 or n_left == n:
        return float("nan")
    if np.all(scores == scores[0]):
        return 0.0
    centered_sum = float(np.sum(scores[left] - np.mean(scores)))
    variance = float(np.var(scores, ddof=1))
    denominator = float(np.sqrt(n_left * (1.0 - n_left / n) * variance))
    magnitude = abs(centered_sum)
    if denominator == 0.0:
        raise ArithmeticError("nonconstant log-rank scores have an unrepresentable variance")
    return magnitude / denominator


def _brier_gradient_data(
    time: FloatArray, event: FloatArray, probability: float
) -> tuple[int, FloatArray | None]:
    """Return RF-SRC's scalar QE event index and its node gradient.

    A zero index deliberately returns an all-zero gradient: the native split
    loop skips that QE point while retaining a zero objective for candidates.
    ``None`` denotes an unusable IPCW point when a required censor-survival
    denominator is zero.
    """
    event_times, failures, at_risk = _parent_counts(time, event)
    if event_times.size == 0:
        return 0, None
    survival = np.cumprod(1.0 - failures / at_risk)
    threshold = 1.0 - probability
    crossing = np.flatnonzero(survival <= threshold)
    selected = int(crossing[0] - 1) if crossing.size else int(event_times.size - 1)
    if selected < 0:
        return 0, np.zeros(time.size, dtype=np.float64)

    censor_times = np.unique(time[event == 0])
    if censor_times.size:
        censor_counts = np.bincount(
            np.searchsorted(censor_times, time[event == 0]), minlength=censor_times.size
        ).astype(np.float64)
        censor_risk = (
            time.size - np.searchsorted(np.sort(time), censor_times, side="left")
        ).astype(np.float64)
        censor_survival = np.cumprod(1.0 - censor_counts / censor_risk)
    else:
        censor_survival = np.empty(0, dtype=np.float64)

    evaluation_time = event_times[selected]
    # C's gHat[0] is the reverse-KM value before the first event. For a
    # selected first event this is G(event_times[0]-), not one by default.
    previous_time = event_times[selected - 1] if selected else event_times[0]

    def censor_survival_before(target: float) -> float:
        position = int(np.searchsorted(censor_times, target, side="left"))
        return 1.0 if position == 0 else float(censor_survival[position - 1])

    g_at_evaluation = censor_survival_before(float(evaluation_time))
    g_at_previous = censor_survival_before(float(previous_time))
    survivors = time > evaluation_time
    failures_by_evaluation = (time <= evaluation_time) & (event == 1)
    if (np.any(survivors) and g_at_evaluation <= 0) or (
        np.any(failures_by_evaluation) and g_at_previous <= 0
    ):
        return selected + 1, None

    weights = np.zeros(time.size, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        weights[survivors] = 1.0 / g_at_evaluation
        weights[failures_by_evaluation] = 1.0 / g_at_previous
        denominator = float(np.sum(weights))
    if not np.isfinite(weights).all() or not np.isfinite(denominator) or denominator <= 0.0:
        return selected + 1, None
    y = survivors.astype(np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        f_hat = float(np.sum(weights * y) / denominator)
        gamma = -2.0 * weights * (y - f_hat)
    if not np.isfinite(gamma).all():
        return selected + 1, None
    return selected + 1, gamma


def _brier_gradient_split_score(gamma: FloatArray | None, left: np.ndarray) -> float:
    """Native global Brier-gradient objective for one candidate partition."""
    if gamma is None:
        return float("nan")
    n = int(gamma.size)
    n_left = int(np.count_nonzero(left))
    n_right = n - n_left
    if n_left == 0 or n_right == 0:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        left_mean = float(np.sum(gamma[left]) / n_left)
        right_mean = float(np.sum(gamma[~left]) / n_right)
        score = (n_left / n) * left_mean**2 + (n_right / n) * right_mean**2
    return score


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
    represented_count: list[int] | None = None,
    imputation_donors: list[tuple[FloatArray, ...]] | None = None,
    training_leaf_node: np.ndarray | None = None,
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
        None
        if represented_count is None
        else _freeze_index(np.asarray(represented_count), np.dtype(np.int32)),
        None
        if imputation_donors is None
        else tuple(tuple(_freeze(values) for values in node) for node in imputation_donors),
        None
        if training_leaf_node is None
        else _freeze_index(training_leaf_node, np.dtype(np.int32)),
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
    split_rule: Literal["logrank", "logrankscore", "bs.gradient", "random"],
    split_probability: float | None,
    retain_represented_count: bool = False,
    original_missing_x: np.ndarray | None = None,
    original_missing_time: np.ndarray | None = None,
    original_missing_event: np.ndarray | None = None,
    master_times: FloatArray | None = None,
    max_imputation_cells: int = _MAX_DESIGN_CELLS,
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
    represented_count = [int(bootstrap_rows.size)] if retain_represented_count else None
    imputation_mode = original_missing_x is not None
    if imputation_mode:
        assert original_missing_x is not None
        assert original_missing_time is not None
        assert original_missing_event is not None
        missing_predictor_mask = original_missing_x
        has_original_missing = bool(
            np.any(original_missing_x)
            or np.any(original_missing_time)
            or np.any(original_missing_event)
        )
    else:
        missing_predictor_mask = None
        has_original_missing = False
    imputation_donors: list[tuple[FloatArray, ...]] | None = (
        [tuple(np.empty(0, dtype=np.float64) for _ in range(x.shape[1]))]
        if imputation_mode
        else None
    )
    if imputation_mode:
        fixed_cells = int(x.size + 4 * x.shape[0])
        if budget.imputation_cells + fixed_cells > max_imputation_cells:
            raise ValueError("forest exceeds max_imputation_cells budget")
        budget.imputation_cells += fixed_cells
    training_leaf_node = np.full(x.shape[0], -1, dtype=np.int32) if imputation_mode else None
    budget.nodes += 1
    if budget.nodes > max_nodes:
        raise ValueError("forest exceeds max_nodes budget")
    feature_count = x.shape[1]
    initial_features = np.ones(feature_count, dtype=bool)
    if imputation_mode:
        assert original_missing_x is not None
        initial_features &= ~np.all(original_missing_x[bootstrap_rows], axis=0)
    initial_all_rows = (
        np.arange(x.shape[0], dtype=np.int64) if imputation_mode else np.empty(0, dtype=np.int64)
    )
    stack: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = [
        (0, bootstrap_rows, initial_all_rows, initial_features, 0)
    ]

    while stack:
        node, rows, all_rows, permissible, depth = stack.pop()
        if imputation_mode:
            assert original_missing_time is not None
            assert original_missing_event is not None
            assert master_times is not None
            node_donors: list[FloatArray] = []
            budget.split_work += int(all_rows.size * (feature_count + 2))
            if budget.split_work > max_split_work:
                raise ValueError("forest exceeds max_split_work budget")
            for column in range(feature_count):
                assert missing_predictor_mask is not None
                donor_rows = rows[~missing_predictor_mask[rows, column]]
                required_cells = int(2 * donor_rows.size + 1)
                if budget.imputation_cells + required_cells > max_imputation_cells:
                    raise ValueError("forest exceeds max_imputation_cells budget")
                donor_values = x[donor_rows, column].copy()
                budget.imputation_cells += required_cells
                node_donors.append(donor_values)
                if np.any(missing_predictor_mask[all_rows, column]):
                    filled, _ = _impute_node_values(
                        x[:, column],
                        missing_predictor_mask[:, column],
                        donor_rows,
                        all_rows,
                        rng=rng,
                    )
                    missing_positions = np.flatnonzero(missing_predictor_mask[all_rows, column])
                    x[all_rows[missing_positions], column] = filled[missing_positions]
            for values, mask, kind in (
                (time, original_missing_time, "time"),
                (event, original_missing_event, "categorical"),
            ):
                donor_rows = rows[~mask[rows]]
                if np.any(mask[all_rows]):
                    filled, _ = _impute_node_values(values, mask, donor_rows, all_rows, rng=rng)
                    missing_positions = np.flatnonzero(mask[all_rows])
                    values[all_rows[missing_positions]] = filled[missing_positions]
            assert imputation_donors is not None
            imputation_donors[node] = tuple(node_donors)
        budget.max_depth = max(budget.max_depth, depth)
        node_time = time[rows]
        node_event = event[rows]
        best_feature = -1
        best_threshold = np.nan
        best_levels: np.ndarray | None = None
        best_score = -np.inf
        next_permissible = permissible.copy()
        stop_time, stop_event = node_time, node_event
        if imputation_mode:
            assert original_missing_time is not None
            assert original_missing_event is not None
            complete_outcomes = ~(original_missing_time[rows] | original_missing_event[rows])
            stop_time = time[rows[complete_outcomes]]
            stop_event = event[rows[complete_outcomes]]
        if stop_time.size >= 2 * nodesize and not _stop_before_split(stop_time, stop_event):
            candidate_features = np.flatnonzero(permissible)
            if candidate_features.size:
                if split_rule == "random":
                    # RF-SRC counts every sampled variable against mtry,
                    # including a variable found constant at this node. It
                    # proceeds to the next draw only when that variable has
                    # no split candidate; the first valid candidate wins.
                    # Its no-missing, uniform-weight fast path enumerates in
                    # stored order only when mtry>1 covers every permissible
                    # feature. mtry==1 remains a random draw even at size one.
                    remaining = candidate_features.copy()
                    ordered_selection = (
                        mtry > 1 and mtry >= remaining.size and not has_original_missing
                    )
                    for _ in range(min(mtry, candidate_features.size)):
                        selected_index = (
                            0 if ordered_selection else int(rng.integers(remaining.size))
                        )
                        column = int(remaining[selected_index])
                        remaining = np.delete(remaining, selected_index)
                        score_rows = rows
                        if imputation_mode:
                            assert original_missing_time is not None
                            assert original_missing_event is not None
                            assert missing_predictor_mask is not None
                            score_rows = rows[
                                ~(
                                    original_missing_time[rows]
                                    | original_missing_event[rows]
                                    | missing_predictor_mask[rows, column]
                                )
                            ]
                            if score_rows.size == 0:
                                next_permissible[column] = False
                                continue
                        budget.split_work += rows.size * ((rows.size - 1).bit_length() + 2)
                        if budget.split_work > max_split_work:
                            raise ValueError("forest exceeds max_split_work budget")
                        unique_values = np.unique(x[score_rows, column])
                        if unique_values.size < 2:
                            next_permissible[column] = False
                            continue
                        if column in categorical_columns:
                            budget.split_work += int(unique_values.size)
                            if budget.split_work > max_split_work:
                                raise ValueError("forest exceeds max_split_work budget")
                            best_levels = _random_factor_partition(unique_values, rng)
                        else:
                            cut_index = int(rng.integers(unique_values.size - 1))
                            best_threshold = float(unique_values[cut_index])
                        best_feature = column
                        best_score = 0.0
                        break
                else:
                    selected = rng.choice(
                        candidate_features,
                        size=min(mtry, candidate_features.size),
                        replace=False,
                    )
                    parent_event_times, parent_events, parent_at_risk = _parent_counts(
                        node_time, node_event
                    )
                    rank_scores = None
                    brier_gamma = None
                    if split_rule == "logrankscore":
                        budget.split_work += rows.size * ((rows.size - 1).bit_length() + 4)
                        if budget.split_work > max_split_work:
                            raise ValueError("forest exceeds max_split_work budget")
                        rank_scores = _logrankscore_scores(node_time, node_event)
                    elif split_rule == "bs.gradient":
                        budget.split_work += rows.size * ((rows.size - 1).bit_length() + 8)
                        if budget.split_work > max_split_work:
                            raise ValueError("forest exceeds max_split_work budget")
                        assert split_probability is not None
                        _, brier_gamma = _brier_gradient_data(
                            node_time, node_event, split_probability
                        )
                    for column in selected:
                        score_rows = rows
                        if imputation_mode:
                            assert original_missing_time is not None
                            assert original_missing_event is not None
                            assert missing_predictor_mask is not None
                            score_rows = rows[
                                ~(
                                    original_missing_time[rows]
                                    | original_missing_event[rows]
                                    | missing_predictor_mask[rows, column]
                                )
                            ]
                            if score_rows.size == 0:
                                next_permissible[column] = False
                                continue
                            budget.split_work += int(
                                score_rows.size * ((score_rows.size - 1).bit_length() + 4)
                            )
                            if budget.split_work > max_split_work:
                                raise ValueError("forest exceeds max_split_work budget")
                            node_time = time[score_rows]
                            node_event = event[score_rows]
                            parent_event_times, parent_events, parent_at_risk = _parent_counts(
                                node_time, node_event
                            )
                            rank_scores = None
                            brier_gamma = None
                            if split_rule == "logrankscore":
                                rank_scores = _logrankscore_scores(node_time, node_event)
                            elif split_rule == "bs.gradient":
                                assert split_probability is not None
                                _, brier_gamma = _brier_gradient_data(
                                    node_time, node_event, split_probability
                                )
                        unique_values = np.unique(x[score_rows, column])
                        if unique_values.size < 2:
                            next_permissible[column] = False
                            continue
                        candidate_iterator: Iterator[tuple[np.ndarray | None, float]]
                        if int(column) in categorical_columns:
                            candidate_count, exact, group_probabilities = _factor_split_plan(
                                unique_values, int(score_rows.size), nsplit
                            )
                            budget.split_work += int(
                                candidate_count * (score_rows.size + unique_values.size)
                            )
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
                            budget.split_work += int(score_rows.size * cuts.size)
                            candidate_iterator = ((None, float(cut)) for cut in cuts)
                        if budget.split_work > max_split_work:
                            raise ValueError("forest exceeds max_split_work budget")
                        for left_levels, cut in candidate_iterator:
                            left_mask = (
                                np.isin(x[score_rows, column], left_levels)
                                if left_levels is not None
                                else x[score_rows, column] <= cut
                            )
                            left_count = int(np.count_nonzero(left_mask))
                            if left_count == 0 or left_count == score_rows.size:
                                continue
                            if split_rule == "logrank":
                                score = _logrank_score(
                                    node_time,
                                    node_event,
                                    left_mask,
                                    parent_events,
                                    parent_at_risk,
                                    parent_event_times,
                                )
                            elif split_rule == "logrankscore":
                                assert rank_scores is not None
                                score = _logrankscore_split_score(rank_scores, left_mask)
                            else:
                                budget.split_work += rows.size
                                if budget.split_work > max_split_work:
                                    raise ValueError("forest exceeds max_split_work budget")
                                score = _brier_gradient_split_score(brier_gamma, left_mask)
                            if split_rule == "bs.gradient" and np.isnan(score):
                                continue
                            if score - best_score > _SPLIT_EPSILON:
                                best_score = score
                                best_feature = int(column)
                                best_threshold = float(cut)
                                best_levels = left_levels

        if best_feature < 0:
            if imputation_mode:
                assert original_missing_time is not None
                assert original_missing_event is not None
                assert master_times is not None
                time_fill = (
                    _terminal_impute_value(
                        time,
                        original_missing_time,
                        rows,
                        rows[original_missing_time[rows]],
                        kind="time",
                        master_times=master_times,
                        rng=rng,
                    )
                    if np.any(original_missing_time[all_rows])
                    else None
                )
                event_fill = (
                    _terminal_impute_value(
                        event,
                        original_missing_event,
                        rows,
                        rows[original_missing_event[rows]],
                        kind="categorical",
                        master_times=master_times,
                        rng=rng,
                    )
                    if np.any(original_missing_event[all_rows])
                    else None
                )
                if time_fill is not None:
                    time[all_rows[original_missing_time[all_rows]]] = time_fill
                if event_fill is not None:
                    event[all_rows[original_missing_event[all_rows]]] = event_fill
                assert training_leaf_node is not None
                training_leaf_node[all_rows] = node
                node_time = time[rows]
                node_event = event[rows]
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
            all_left_mask = (
                np.isin(x[all_rows, best_feature], best_levels)
                if imputation_mode
                else np.empty(0, dtype=bool)
            )
        else:
            left_mask = x[rows, best_feature] <= best_threshold
            all_left_mask = (
                x[all_rows, best_feature] <= best_threshold
                if imputation_mode
                else np.empty(0, dtype=bool)
            )
        left_rows = rows[left_mask]
        right_rows = rows[~left_mask]
        left_all_rows = all_rows[all_left_mask] if imputation_mode else all_rows
        right_all_rows = all_rows[~all_left_mask] if imputation_mode else all_rows
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
            if represented_count is not None:
                represented_count.append(int(left_rows.size) if _ == 0 else int(right_rows.size))
            if imputation_donors is not None:
                if budget.imputation_cells + feature_count > max_imputation_cells:
                    raise ValueError("forest exceeds max_imputation_cells budget")
                budget.imputation_cells += feature_count
                imputation_donors.append(
                    tuple(np.empty(0, dtype=np.float64) for _ in range(feature_count))
                )
        # Native trees process the left branch first; preserve that random
        # draw order by pushing right before left on this LIFO work list.
        stack.append((right_index, right_rows, right_all_rows, next_permissible, depth + 1))
        stack.append((left_index, left_rows, left_all_rows, next_permissible, depth + 1))

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
        represented_count,
        imputation_donors,
        training_leaf_node,
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
    row_indices: np.ndarray | None = None,
    compute_concordance: bool = True,
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
            if tree.training_leaf_node is not None:
                node = int(tree.training_leaf_node[row])
                if node < 0:
                    raise RuntimeError("imputed training route is missing a terminal node")
            else:
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
    if compute_concordance:
        concordance_error, comparable_pairs = _oob_concordance_error(
            time, event, mortality, contributors
        )
    else:
        concordance_error, comparable_pairs = float("nan"), 0
    return RandomSurvivalForestOOB(
        _freeze(time_grid),
        _freeze(survival),
        _freeze(cumulative_hazard),
        np.frombuffer(contributors.tobytes(), dtype=np.int64),
        _freeze(mortality),
        concordance_error,
        comparable_pairs,
        _freeze_index(
            np.arange(n, dtype=np.int64) if row_indices is None else row_indices,
            np.dtype(np.int64),
        ),
        compute_concordance,
    )


def fit_random_survival_forest(
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    categorical_features: ArrayLike | None = None,
    split_rule: Literal["logrank", "logrankscore", "bs.gradient", "random"] = "logrank",
    prob: float | None = None,
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
    na_action: Literal["raise", "omit", "impute"] = "raise",
    max_imputation_cells: int = _MAX_IMPUTATION_CELLS,
) -> RandomSurvivalForestFit:
    """Fit right-censored survival trees with numeric/nominal features.

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
    Set ``compute_oob=True`` to retain bit-packed in-bag membership, per-node
    bootstrap represented counts, and OOB curves, mortality and concordance
    error; the default avoids this storage and work. Rows without an OOB tree
    remain undefined (NaN), never in-bag-filled.

    ``split_rule="logrank"`` preserves the default source-pinned score.
    ``split_rule="logrankscore"`` uses the published Hothorn--Lausen
    standardized rank scores with maximum-rank time ties. This corrects
    indexing defects identified in RF-SRC's optional ``SURV_LRSCR`` branch;
    it does not claim numerical parity with that branch.
    ``split_rule="bs.gradient"`` uses RF-SRC 3.2.2's one-point global Brier
    score gradient. ``prob`` is its scalar failure-quantile probability
    (default 0.9); it selects one prior event-grid point using the parent
    Kaplan--Meier curve's ``1 - prob`` survival threshold.
    ``split_rule="random"`` draws one cut/partition on the first selected
    feature with a valid split; it does not compare survival scores.

    ``na_action="omit"`` fits complete rows and retains their original indices.
    ``na_action="impute"`` uses one pass of node-local observed in-bag donors,
    while original missingness excludes rows from candidate-specific scores.
    Donor-less outcome bootstraps are skipped; ``requested_trees`` records the
    attempted count. ``max_imputation_cells`` bounds retained donors and work
    arrays. Missing-outcome OOB concordance is explicitly unavailable.
    """
    if not isinstance(replace, (bool, np.bool_)):
        raise ValueError("replace must be boolean")
    if not isinstance(compute_oob, (bool, np.bool_)):
        raise ValueError("compute_oob must be boolean")
    if not isinstance(split_rule, str) or split_rule not in _SPLIT_RULES:
        raise ValueError("split_rule must be 'logrank', 'logrankscore', 'bs.gradient', or 'random'")
    if split_rule == "bs.gradient":
        raw_prob = _DEFAULT_BRIER_PROB if prob is None else prob
        if isinstance(raw_prob, (bool, np.bool_)) or np.iscomplexobj(raw_prob):
            raise ValueError("prob must be a real number strictly between 0 and 1")
        split_probability = scalar(raw_prob, "prob")
        if not 0.0 < split_probability < 1.0:
            raise ValueError("prob must be strictly between 0 and 1")
    else:
        if prob is not None:
            raise ValueError("prob is only available with split_rule='bs.gradient'")
        split_probability = None
    oob_cell_limit = _budget_limit(max_oob_cells, "max_oob_cells", _MAX_OOB_CELLS, _MAX_OOB_CELLS)
    oob_work_limit = _budget_limit(max_oob_work, "max_oob_work", _MAX_OOB_WORK, _MAX_OOB_WORK)
    tree_count = _integer(n_trees, "n_trees", 1, _MAX_TREES)
    leaf_size = _integer(nodesize, "nodesize", 1, _MAX_ROWS)
    random_splits = _integer(nsplit, "nsplit", 0, _MAX_ROWS)
    if na_action not in ("raise", "omit", "impute"):
        raise ValueError("na_action must be 'raise', 'omit', or 'impute'")
    original_missing_x: np.ndarray | None = None
    original_missing_time: np.ndarray | None = None
    original_missing_event: np.ndarray | None = None
    master_times: FloatArray | None = None
    if na_action == "raise":
        t, e, raw_x = _forest_data(time, event, covariates)
        row_indices = np.arange(t.size, dtype=np.int64)
        original_fingerprint = None
        imputation_performed = False
        missing_x = np.zeros(raw_x.shape, dtype=bool)
    else:
        original_t, original_e, original_x = _forest_data_allow_missing(time, event, covariates)
        if np.all(np.isnan(original_t)) or np.all(np.isnan(original_e)):
            raise ValueError("na_action cannot recover an entirely missing outcome variable")
        if original_x.shape[1] and np.any(np.all(np.isnan(original_x), axis=0)):
            raise ValueError("na_action cannot recover an entirely missing predictor column")
        all_missing = np.isnan(original_t) & np.isnan(original_e)
        if original_x.shape[1]:
            all_missing &= np.all(np.isnan(original_x), axis=1)
        active_rows = np.flatnonzero(~all_missing).astype(np.int64, copy=False)
        if na_action == "omit":
            complete = ~np.isnan(original_t[active_rows]) & ~np.isnan(original_e[active_rows])
            if original_x.shape[1]:
                complete &= np.all(np.isfinite(original_x[active_rows]), axis=1)
            row_indices = active_rows[complete]
        else:
            row_indices = active_rows
            original_missing_x = np.isnan(original_x[row_indices])
            original_missing_time = np.isnan(original_t[row_indices])
            original_missing_event = np.isnan(original_e[row_indices])
            master_times = np.unique(original_t[np.isfinite(original_t)])
            if not master_times.size:
                raise ValueError("imputation requires at least one observed time")
        original_fingerprint = _forest_original_fingerprint(original_t, original_e, original_x)
        if not row_indices.size:
            raise ValueError("na_action removed every training row")
        if na_action == "impute":
            imputation_performed = bool(
                np.any(original_missing_x)
                or np.any(original_missing_time)
                or np.any(original_missing_event)
            )
            t = original_t[row_indices].copy()
            e = original_e[row_indices].copy()
            raw_x = original_x[row_indices].copy()
            t[np.isnan(t)] = 0.0
            e[np.isnan(e)] = 0.0
            raw_x[np.isnan(raw_x)] = 0.0
        else:
            t = original_t[row_indices].copy()
            e = original_e[row_indices].copy()
            raw_x = original_x[row_indices].copy()
            imputation_performed = False
            missing_x = np.zeros(raw_x.shape, dtype=bool)
    if na_action == "impute":
        assert original_missing_x is not None
        x, categorical_levels, categorical = _categorical_columns(
            raw_x, categorical_features, original_missing_x
        )
        complete_events = np.isfinite(original_t) & np.isfinite(original_e) & (original_e == 1)
        event_times = np.unique(original_t[complete_events])
        if not event_times.size:
            raise ValueError(
                "imputation requires an observed complete event to define the output grid"
            )
        output_grid = _time_grid(event_times, ntime)
        raw_x_for_profile = original_x[row_indices]
        original_missing_x_for_profile = original_missing_x
    else:
        x, categorical_levels, categorical = _categorical_columns(raw_x, categorical_features)
        event_times = np.unique(t[e == 1])
        output_grid = _time_grid(event_times, ntime)
        raw_x_for_profile = raw_x
        original_missing_x_for_profile = missing_x
    if original_fingerprint is not None:
        original_fingerprint = _forest_original_fingerprint(
            original_t, original_e, original_x, categorical
        )
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
    imputation_limit = _budget_limit(
        max_imputation_cells,
        "max_imputation_cells",
        _MAX_IMPUTATION_CELLS,
        _MAX_IMPUTATION_CELLS,
    )
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
    if na_action == "impute":
        minimum_imputation_cells = tree_count * (
            x.size + 4 * t.size + x.shape[1] * (2 * sample_size + 1)
        )
        if minimum_imputation_cells > imputation_limit:
            raise ValueError(
                "imputation work arrays and root donors exceed max_imputation_cells budget"
            )
    if na_action != "impute":
        event_times = np.unique(t[e == 1])
        output_grid = _time_grid(event_times, ntime)
    membership: np.ndarray | None = None
    if compute_oob:
        packed_width = (t.size + 7) // 8
        membership_cells = tree_count * packed_width
        output_cells = t.size * output_grid.size
        combined_cells = 2 * membership_cells + 8 * output_cells + 16 * t.size
        node_budget = min(node_limit, oob_cell_limit - combined_cells)
        if node_budget < tree_count:
            raise ValueError("OOB membership, curves, and node counts exceed max_oob_cells budget")
        node_limit = node_budget
        if t.size * t.size > oob_work_limit:
            raise ValueError("OOB pairwise-concordance work exceeds max_oob_work budget")
        membership = np.zeros((tree_count, packed_width), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    budget = _Budget()
    trees: list[_PackedTree] = []
    effective_membership: list[np.ndarray] = []
    for tree_index in range(tree_count):
        bootstrap = rng.choice(t.size, size=sample_size, replace=bool(replace))
        if na_action == "impute":
            assert original_missing_time is not None
            assert original_missing_event is not None
            if np.all(original_missing_time[bootstrap]) or np.all(
                original_missing_event[bootstrap]
            ):
                continue
        if membership is not None:
            inbag = np.zeros(t.size, dtype=np.uint8)
            inbag[bootstrap] = 1
            packed_inbag = np.packbits(inbag, bitorder="little")
            if na_action == "impute":
                effective_membership.append(packed_inbag)
            else:
                membership[tree_index] = packed_inbag
        if na_action == "impute":
            tree_x, tree_t, tree_e = x.copy(), t.copy(), e.copy()
        else:
            tree_x, tree_t, tree_e = x, t, e
        trees.append(
            _grow_tree(
                tree_x,
                tree_t,
                tree_e,
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
                split_rule=split_rule,
                split_probability=split_probability,
                retain_represented_count=bool(compute_oob),
                original_missing_x=original_missing_x,
                original_missing_time=original_missing_time,
                original_missing_event=original_missing_event,
                master_times=master_times,
                max_imputation_cells=imputation_limit,
            )
        )
    if not trees:
        raise ValueError("no bootstrap replicate had observed donors for every outcome variable")
    if na_action == "impute":
        sampled_rows = len(trees) * sample_size
        if membership is not None:
            membership = np.asarray(effective_membership, dtype=np.uint8)
    packed_membership = (
        None
        if membership is None
        else np.frombuffer(membership.tobytes(), dtype=np.uint8).reshape(membership.shape)
    )
    missing_outcomes = False
    if na_action == "impute":
        assert original_missing_time is not None
        assert original_missing_event is not None
        missing_outcomes = bool(np.any(original_missing_time) or np.any(original_missing_event))
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
            row_indices=row_indices,
            compute_concordance=not missing_outcomes,
        )
    )
    return RandomSurvivalForestFit(
        _freeze(output_grid),
        _freeze(
            _modal_profile_observed(
                raw_x_for_profile, original_missing_x_for_profile, categorical_levels
            )
            if na_action == "impute"
            else _modal_profile(raw_x, categorical_levels)
        ),
        x.shape[1],
        tuple(trees),
        len(trees),
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
        split_rule,
        split_probability,
        na_action,
        imputation_performed,
        _freeze_index(row_indices, np.dtype(np.int64)),
        original_fingerprint,
        tree_count,
    )


def predict_random_survival_forest(
    fit: RandomSurvivalForestFit,
    times: ArrayLike | None = None,
    profiles: ArrayLike | None = None,
    *,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
    max_prediction_work: int = _MAX_PREDICTION_WORK,
    random_state: int | None = None,
    na_action: Literal["raise", "omit", "impute"] = "raise",
) -> RandomSurvivalForestPrediction:
    """Predict the separately averaged survival and cumulative-hazard curves.

    ``na_action="omit"`` removes incomplete profiles and records their original
    positions. ``na_action="impute"`` requires retained training donors and an
    explicit ``random_state`` when a profile has missing values.
    """
    if na_action not in ("raise", "omit", "impute"):
        raise ValueError("prediction na_action must be 'raise', 'omit', or 'impute'")
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
        profile_missing = np.zeros(profile_values.shape, dtype=bool)
    else:
        _preflight_profile_matrix(profiles)
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile_values = np.asarray(profiles, dtype=np.float64)
        if profile_values.ndim == 1:
            profile_values = profile_values[None, :]
        if np.any(np.isinf(profile_values)):
            raise ValueError(
                "profiles may contain finite values or NaN missing values, not infinity"
            )
        profile_missing = np.isnan(profile_values)
    if (
        profile_values.ndim != 2
        or profile_values.shape[1] != fit.covariate_count
        or profile_values.shape[0] == 0
        or profile_values.shape[0] > 100_000
        or profile_values.size > _MAX_DESIGN_CELLS
    ):
        raise ValueError("profiles must have one column per fitted covariate")
    row_indices = np.arange(profile_values.shape[0], dtype=np.int64)
    if na_action == "omit" and np.any(profile_missing):
        row_indices = np.flatnonzero(~np.any(profile_missing, axis=1))
        if not row_indices.size:
            raise ValueError("na_action removed every prediction profile")
        profile_values = profile_values[row_indices]
        profile_missing = np.zeros(profile_values.shape, dtype=bool)
    has_missing_profiles = bool(np.any(profile_missing))
    if has_missing_profiles:
        if na_action != "impute":
            raise ValueError("missing profiles require prediction na_action='impute'")
        if fit.na_action != "impute" or not any(
            tree.imputation_donors is not None for tree in fit.trees
        ):
            raise ValueError("missing profiles require a fit created with na_action='impute'")
        if random_state is None:
            raise ValueError("random_state is required to impute missing prediction profiles")
    if random_state is not None:
        prediction_seed = _integer(random_state, "random_state", 0, np.iinfo(np.int32).max)
        prediction_rng = np.random.default_rng(prediction_seed)
    else:
        prediction_rng = None
    categorical_levels = fit.categorical_levels or tuple(None for _ in range(fit.covariate_count))
    output_limit = _integer(max_output_cells, "max_output_cells", 1, _MAX_OUTPUT_CELLS)
    work_limit = _integer(max_prediction_work, "max_prediction_work", 1, _MAX_PREDICTION_WORK)
    cells = int(profile_values.shape[0] * time_values.size)
    categorical_copy = (
        profile_values.size if any(levels is not None for levels in categorical_levels) else 0
    )
    combined_cells = 8 * cells + profile_values.size + time_values.size + categorical_copy
    if has_missing_profiles:
        combined_cells += 2 * profile_values.size
    if combined_cells > output_limit:
        raise ValueError("prediction exceeds max_output_cells budget")
    work = cells * fit.n_trees + profile_values.shape[0] * fit.n_trees * fit.max_depth
    if has_missing_profiles:
        work += int(np.count_nonzero(profile_missing)) * fit.n_trees * (fit.max_depth + 1)
    if work > work_limit:
        raise ValueError("prediction exceeds max_prediction_work budget")
    routed_profiles = _encode_profiles(profile_values, categorical_levels, profile_missing)
    if has_missing_profiles:
        routed_profiles = routed_profiles.copy()
        routed_profiles[profile_missing] = 0.0
    log_survival_sum = np.full((profile_values.shape[0], time_values.size), -np.inf)
    hazard_sum = np.zeros_like(log_survival_sum)
    for tree in fit.trees:
        for profile_index, original_profile in enumerate(routed_profiles):
            profile = original_profile.copy() if has_missing_profiles else original_profile
            node = 0
            while tree.feature[node] >= 0:
                if has_missing_profiles:
                    assert prediction_rng is not None
                    donors_by_node = tree.imputation_donors
                    if donors_by_node is None:
                        raise ValueError("fit lacks node-local predictor donors for imputation")
                    missing_columns = np.flatnonzero(profile_missing[profile_index])
                    for missing_column in missing_columns:
                        feature_index = int(missing_column)
                        donor_pool = donors_by_node[node][feature_index]
                        if donor_pool.size:
                            profile[feature_index] = donor_pool[
                                int(prediction_rng.integers(donor_pool.size))
                            ]
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
        _freeze_index(row_indices, np.dtype(np.int64)),
    )
