"""Bounded pooling of per-tree values for iterated missing-data completion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ._cdflib import _freeze
from ._validation import FloatArray

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
_MAX_POOL_CELLS = 2_000_000
_MAX_POOL_WORK = 100_000_000


class _UniformSource(Protocol):
    def random(self) -> float: ...


class _TreeSummary(Protocol):
    @property
    def training_leaf_node(self) -> np.ndarray | None: ...

    @property
    def terminal_time(self) -> FloatArray | None: ...

    @property
    def terminal_event(self) -> FloatArray | None: ...

    @property
    def terminal_predictor(self) -> FloatArray | None: ...


@dataclass(frozen=True)
class ImputationPoolResult:
    """Completed arrays and cells that required a global donor fallback.

    ``fallback_used`` has columns for time, event status, and then predictors.
    The returned arrays are owned and read-only.  The helper's generator is a
    Python reproducibility convention; it does not reproduce randomForestSRC's
    native random streams.
    """

    time: FloatArray
    event: FloatArray
    covariates: FloatArray
    fallback_used: BoolArray
    work_units: int


def _positive_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1 or result > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum:,}")
    return result


def _uniform(rng: _UniformSource) -> float:
    value = float(rng.random())
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("rng.random() must return a finite value in [0, 1)")
    return value


def _owned_bool_mask(values: BoolArray, shape: tuple[int, ...], name: str) -> BoolArray:
    if (
        not isinstance(values, np.ndarray)
        or values.shape != shape
        or values.dtype != np.dtype(bool)
    ):
        raise ValueError(f"{name} must be a Boolean array with shape {shape}")
    return values


def _scaled_mean_add(
    scale: FloatArray,
    scaled_sum: FloatArray,
    count: IntArray,
    rows: NDArray[np.int64],
    values: FloatArray,
) -> None:
    """Add finite values by row without overflowing an ordinary sum."""
    if not rows.size:
        return
    old_scale = scale[rows]
    new_scale = np.maximum(old_scale, np.abs(values))
    old_scaled = np.zeros(rows.size, dtype=np.float64)
    positive = old_scale > 0
    old_scaled[positive] = scaled_sum[rows[positive]] * (old_scale[positive] / new_scale[positive])
    new_scaled = np.zeros(rows.size, dtype=np.float64)
    nonzero = new_scale > 0
    new_scaled[nonzero] = values[nonzero] / new_scale[nonzero]
    scale[rows] = new_scale
    scaled_sum[rows] = old_scaled + new_scaled
    count[rows] += 1


def _scaled_mean(
    scale: FloatArray, scaled_sum: FloatArray, count: IntArray, rows: NDArray[np.int64]
) -> FloatArray:
    answer = np.zeros(rows.size, dtype=np.float64)
    has_values = count[rows] > 0
    selected = rows[has_values]
    if selected.size:
        answer[has_values] = scale[selected] * (scaled_sum[selected] / count[selected])
    if not np.isfinite(answer).all():
        raise ArithmeticError("pooled numeric imputation is not finite")
    return answer


def _nearest_master_time(
    values: FloatArray, master_times: FloatArray, rng: _UniformSource
) -> FloatArray:
    positions = np.searchsorted(master_times, values, side="left")
    right = np.minimum(positions, master_times.size - 1)
    left = np.maximum(positions - 1, 0)
    left_distance = np.abs(values - master_times[left])
    right_distance = np.abs(master_times[right] - values)
    difference = left_distance - right_distance
    choose_left = (left != right) & (left_distance < right_distance)
    result = master_times[np.where(choose_left, left, right)].copy()
    ties = (left != right) & ~choose_left & (np.abs(difference) < 1e-9)
    for row in np.flatnonzero(ties):
        result[row] = master_times[int(left[row] if _uniform(rng) <= 0.5 else right[row])]
    return result


def pool_imputation_summaries(
    time: FloatArray,
    event: FloatArray,
    covariates: FloatArray,
    missing_time: BoolArray,
    missing_event: BoolArray,
    missing_covariates: BoolArray,
    trees: Sequence[_TreeSummary],
    membership: NDArray[np.uint8],
    *,
    master_times: FloatArray,
    rng: _UniformSource,
    categorical_features: Sequence[int] = (),
    selection: Literal["oob", "all"] = "oob",
    max_cells: int = _MAX_POOL_CELLS,
    max_work: int = _MAX_POOL_WORK,
) -> ImputationPoolResult:
    """Pool tree terminal imputations, then use original observed donors.

    Each tree contributes at most one scalar per missing row and field.  Numeric
    fields use a scaled arithmetic mean; status and categorical predictors use
    the mode with uniform random tie-breaking.  If no tree contributes to a
    cell, a donor is sampled uniformly from that field's original observed
    values.  Time values are snapped to the fixed master grid after both
    pooling and fallback.

    ``membership`` stores one little-bit-order in-bag mask per tree.  ``oob``
    uses rows whose bit is zero; ``all`` includes every row for every tree.
    Original missing masks remain authoritative even though input arrays are
    completed values from the preceding pass.

    The reported work charge is a conservative input-derived bound: tree and
    row/field visits, plus category level scans and mode resolution.
    """
    if not isinstance(time, np.ndarray) or time.ndim != 1 or time.dtype.kind not in "iuf":
        raise ValueError("time must be a real one-dimensional array")
    if not isinstance(event, np.ndarray) or event.ndim != 1 or event.dtype.kind not in "iuf":
        raise ValueError("event must be a real one-dimensional array")
    if (
        not isinstance(covariates, np.ndarray)
        or covariates.ndim != 2
        or covariates.dtype.kind not in "iuf"
    ):
        raise ValueError("covariates must be a real two-dimensional array")
    n = int(time.size)
    p = int(covariates.shape[1])
    if n < 1 or event.shape != (n,) or covariates.shape[0] != n:
        raise ValueError("time, event, and covariates must have the same positive row count")
    missing_time = _owned_bool_mask(missing_time, (n,), "missing_time")
    missing_event = _owned_bool_mask(missing_event, (n,), "missing_event")
    missing_covariates = _owned_bool_mask(missing_covariates, (n, p), "missing_covariates")
    if selection not in ("oob", "all"):
        raise ValueError("selection must be 'oob' or 'all'")
    if not callable(getattr(rng, "random", None)):
        raise ValueError("rng must provide random() uniform draws")
    cell_limit = _positive_integer(max_cells, "max_cells", _MAX_POOL_CELLS)
    work_limit = _positive_integer(max_work, "max_work", _MAX_POOL_WORK)

    # Check metadata and conservative workspace/work limits before making any
    # proportional copies.  Streaming tree-by-tree avoids a trees×rows×fields cube.
    expected_mask_bytes = (n + 7) // 8
    if not isinstance(membership, np.ndarray) or membership.dtype != np.dtype(np.uint8):
        raise ValueError("membership must be a uint8 packed in-bag matrix")
    if membership.shape != (len(trees), expected_mask_bytes):
        raise ValueError("membership must align with trees and training rows")
    if not isinstance(master_times, np.ndarray) or master_times.ndim != 1 or master_times.size < 1:
        raise ValueError("master_times must be a nonempty one-dimensional array")
    if master_times.dtype.kind not in "iuf" or not np.isfinite(master_times).all():
        raise ValueError("master_times must contain finite real values")
    if np.any(master_times < 0.0) or np.any(np.diff(master_times) <= 0):
        raise ValueError("master_times must be strictly increasing")
    if (
        not np.isfinite(time).all()
        or not np.isfinite(event).all()
        or not np.isfinite(covariates).all()
    ):
        raise ValueError("completed inputs must be finite")
    if np.any((event != 0.0) & (event != 1.0)):
        raise ValueError("event values must be zero or one")
    if np.any(time < 0.0):
        raise ValueError("time values must be nonnegative")
    if any(
        isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer))
        for j in categorical_features
    ):
        raise ValueError("categorical_features must contain integer indices")
    feature_set = {int(j) for j in categorical_features}
    if any(int(j) < 0 or int(j) >= p for j in feature_set):
        raise ValueError("categorical feature index is out of range")

    n_fields = p + 2
    workspace_cells = 9 * n * n_fields + int(master_times.size) + 2 * len(trees) + 1
    if workspace_cells > cell_limit:
        raise ValueError("imputation pooling exceeds max_cells budget")
    missing_rows = np.flatnonzero(missing_time | missing_event | np.any(missing_covariates, axis=1))
    if not missing_rows.size:
        return ImputationPoolResult(
            _freeze(time.copy()),
            _freeze(event.copy()),
            _freeze(covariates.copy()),
            np.frombuffer(np.zeros((n, n_fields), dtype=bool).tobytes(), dtype=np.bool_).reshape(
                n, n_fields
            ),
            0,
        )

    active_categorical = {
        feature for feature in feature_set if np.any(missing_covariates[:, feature])
    }
    category_levels: dict[int, FloatArray] = {}
    category_cells = 0
    category_resolution_work = 0
    for feature in active_categorical:
        levels = np.unique(covariates[~missing_covariates[:, feature], feature])
        if not levels.size:
            raise ValueError(f"categorical predictor {feature} has no original observed donor")
        category_levels[feature] = levels
        category_cells += n * int(levels.size) + int(levels.size)
        category_resolution_work += n * int(levels.size)
        if workspace_cells + category_cells > cell_limit:
            raise ValueError("categorical imputation workspace exceeds max_cells budget")
    work = (len(trees) + 1) * n * n_fields
    work += category_resolution_work + n * max(1, n.bit_length()) * len(active_categorical)
    if work > work_limit:
        raise ValueError("imputation pooling exceeds max_work budget")

    completed_time = time.copy()
    completed_event = event.copy()
    completed_x = covariates.copy()
    fallback = np.zeros((n, n_fields), dtype=bool)
    # Stable mean sufficient statistics for continuous fields.
    time_scale = np.zeros(n, dtype=np.float64)
    time_sum = np.zeros(n, dtype=np.float64)
    time_count = np.zeros(n, dtype=np.int64)
    x_scale = np.zeros((n, p), dtype=np.float64)
    x_sum = np.zeros((n, p), dtype=np.float64)
    x_count = np.zeros((n, p), dtype=np.int64)
    event_counts = np.zeros((n, 2), dtype=np.int32)

    category_counts: dict[int, NDArray[np.int32]] = {}
    for feature in active_categorical:
        levels = category_levels[feature]
        category_levels[feature] = levels
        category_counts[feature] = np.zeros((n, levels.size), dtype=np.int32)

    for tree_index, tree in enumerate(trees):
        leaves = getattr(tree, "training_leaf_node", None)
        terminal_time = getattr(tree, "terminal_time", None)
        terminal_event = getattr(tree, "terminal_event", None)
        terminal_x = getattr(tree, "terminal_predictor", None)
        if leaves is None or leaves.shape != (n,):
            raise RuntimeError("tree lacks training leaf assignments for imputation pooling")
        if selection == "oob":
            inbag = np.unpackbits(membership[tree_index], bitorder="little")[:n].astype(bool)
            eligible = ~inbag
        else:
            eligible = np.ones(n, dtype=bool)
        rows = np.flatnonzero(
            eligible & (missing_time | missing_event | np.any(missing_covariates, axis=1))
        )
        if not rows.size:
            continue
        nodes = leaves[rows]
        if np.any(nodes < 0):
            raise RuntimeError("tree leaf assignment is missing for an eligible imputation row")
        if terminal_time is not None and (
            terminal_time.ndim != 1 or np.any(nodes >= terminal_time.size)
        ):
            raise RuntimeError("tree terminal summary index is out of range")
        if terminal_event is not None and (
            terminal_event.ndim != 1 or np.any(nodes >= terminal_event.size)
        ):
            raise RuntimeError("tree terminal status summary index is out of range")
        if terminal_x is not None and (
            terminal_x.ndim != 2 or terminal_x.shape[1] != p or np.any(nodes >= terminal_x.shape[0])
        ):
            raise RuntimeError("tree terminal predictor summaries have an incompatible shape")

        if terminal_time is not None:
            selected = missing_time[rows]
            selected_rows = rows[selected]
            selected_values = terminal_time[nodes[selected]]
            valid = np.isfinite(selected_values)
            _scaled_mean_add(
                time_scale,
                time_sum,
                time_count,
                selected_rows[valid],
                selected_values[valid],
            )
        if terminal_event is not None:
            selected = missing_event[rows]
            selected_rows = rows[selected]
            selected_values = terminal_event[nodes[selected]]
            for value in (0.0, 1.0):
                value_rows = selected_rows[selected_values == value]
                event_counts[value_rows, int(value)] += 1
        if terminal_x is not None:
            for feature_index in np.flatnonzero(np.any(missing_covariates[rows], axis=0)):
                feature = int(feature_index)
                selected = missing_covariates[rows, feature]
                selected_rows = rows[selected]
                selected_values = terminal_x[nodes[selected], feature]
                valid = np.isfinite(selected_values)
                selected_rows = selected_rows[valid]
                selected_values = selected_values[valid]
                if not selected_rows.size:
                    continue
                if int(feature) in active_categorical:
                    levels = category_levels[int(feature)]
                    level_indices = np.searchsorted(levels, selected_values)
                    if np.any(level_indices >= levels.size) or np.any(
                        levels[level_indices] != selected_values
                    ):
                        raise RuntimeError(
                            "categorical terminal summary is not an original observed level"
                        )
                    np.add.at(category_counts[int(feature)], (selected_rows, level_indices), 1)
                else:
                    _scaled_mean_add(
                        x_scale[:, feature],
                        x_sum[:, feature],
                        x_count[:, feature],
                        selected_rows,
                        selected_values,
                    )

    # Resolve local summaries in row-major field order so random mode ties
    # consume draws before the later variable-major global fallback phase.
    for row_index in range(n):
        row = row_index
        if missing_time[row] and time_count[row] > 0:
            completed_time[row] = _scaled_mean(
                time_scale, time_sum, time_count, np.asarray([row], dtype=np.int64)
            )[0]
        if missing_event[row] and event_counts[row].sum() > 0:
            maximum = int(event_counts[row].max())
            candidates = np.flatnonzero(event_counts[row] == maximum)
            completed_event[row] = float(
                candidates[0]
                if candidates.size == 1
                else candidates[min(int(_uniform(rng) * candidates.size), candidates.size - 1)]
            )
        for feature in range(p):
            if not missing_covariates[row, feature]:
                continue
            if feature in active_categorical:
                counts = category_counts[feature][row]
                if counts.sum() > 0:
                    maximum = int(counts.max())
                    candidates = np.flatnonzero(counts == maximum)
                    choice = int(
                        candidates[0]
                        if candidates.size == 1
                        else candidates[
                            min(int(_uniform(rng) * candidates.size), candidates.size - 1)
                        ]
                    )
                    completed_x[row, feature] = category_levels[feature][choice]
            elif x_count[row, feature] > 0:
                completed_x[row, feature] = _scaled_mean(
                    x_scale[:, feature],
                    x_sum[:, feature],
                    x_count[:, feature],
                    np.asarray([row], dtype=np.int64),
                )[0]

    # Native fallback is an observed-value donor draw, in variable-major order,
    # after all local mode ties have consumed random numbers.
    donor_fields: list[tuple[int, np.ndarray, FloatArray]] = [
        (0, missing_time, time[~missing_time]),
        (1, missing_event, event[~missing_event]),
    ]
    donor_fields.extend(
        (
            feature + 2,
            missing_covariates[:, feature],
            covariates[~missing_covariates[:, feature], feature],
        )
        for feature in range(p)
    )
    for field, missing, donors in donor_fields:
        rows = np.flatnonzero(missing)
        if field == 0:
            needs_fallback = rows[time_count[rows] == 0]
        elif field == 1:
            needs_fallback = rows[event_counts[rows].sum(axis=1) == 0]
        elif field - 2 in active_categorical:
            counts = category_counts[field - 2]
            needs_fallback = rows[counts[rows].sum(axis=1) == 0]
        else:
            feature = field - 2
            needs_fallback = rows[x_count[rows, feature] == 0]
        if needs_fallback.size and not donors.size:
            raise ValueError(f"field {field} has missing values but no original observed donor")
        for row in needs_fallback:
            donor = donors[min(int(_uniform(rng) * donors.size), donors.size - 1)]
            if field == 0:
                completed_time[row] = float(donor)
            elif field == 1:
                completed_event[row] = float(donor)
            else:
                completed_x[row, field - 2] = float(donor)
            fallback[row, field] = True

    missing_time_rows = np.flatnonzero(missing_time)
    if missing_time_rows.size:
        completed_time[missing_time_rows] = _nearest_master_time(
            completed_time[missing_time_rows], master_times, rng
        )
    if (
        not np.isfinite(completed_time).all()
        or not np.isfinite(completed_event).all()
        or not np.isfinite(completed_x).all()
    ):
        raise ArithmeticError("imputation pooling produced nonfinite values")
    if np.any((completed_event != 0.0) & (completed_event != 1.0)):
        raise ArithmeticError("imputation pooling produced an invalid event status")
    return ImputationPoolResult(
        _freeze(completed_time),
        _freeze(completed_event),
        _freeze(completed_x),
        np.frombuffer(fallback.tobytes(), dtype=np.bool_).reshape(fallback.shape),
        work,
    )
