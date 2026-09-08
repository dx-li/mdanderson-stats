"""MUHAZ failure-count and survival-mass nearest-neighbor bandwidths."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite

NeighborMethod = Literal["failures", "survival"]


@dataclass(frozen=True)
class NeighborBandwidths:
    time: FloatArray
    neighbors: NDArray[np.int64]
    bandwidth: FloatArray
    method: NeighborMethod
    legacy: bool


def muhaz_neighbor_bandwidths(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    neighbors: ArrayLike,
    grid: ArrayLike,
    method: NeighborMethod = "survival",
    legacy: bool = False,
) -> NeighborBandwidths:
    """Compute bandwidths (neighbor-count rows, evaluation-time columns).

    Failures mode returns kth absolute distance to an observed failure. Survival
    mode applies the archived survival-mass radius rule, including its 1e-5
    perturbations and index window. Legacy survival mode also omits a final
    singleton observation, matching KAPMEI's loop-bound error.

    Zero radii are retained (e.g. tied failures); a hazard fit requires positive
    bandwidths. Negative or nonfinite source outcomes raise RuntimeError.
    """
    t = finite(times, "times")
    if t.ndim != 1 or t.size == 0 or np.any(t < 0):
        raise ValueError("times must be a nonempty nonnegative vector")
    d = np.ones(t.size) if delta is None else count(delta, "delta")
    if d.shape != t.shape or np.any(d > 1):
        raise ValueError("delta must match times and contain only 0 or 1")
    z = finite(grid, "grid").copy()
    if z.ndim != 1 or np.any(z < 0):
        raise ValueError("grid must be a nonnegative vector")
    if method not in ("failures", "survival"):
        raise ValueError("method must be failures or survival")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    n_events = int(d.sum())
    if n_events == 0:
        raise ValueError("Nearest-neighbor bandwidths require observed failures")
    counts = count(neighbors, "neighbors")
    limit = n_events if method == "failures" else t.size
    if counts.ndim > 1 or counts.size == 0 or np.any((counts < 1) | (counts > limit)):
        raise ValueError("neighbors must be a positive scalar or vector within the sample limit")
    k = counts.reshape(-1).astype(np.int64)
    order = np.argsort(t, kind="stable")
    t, d = t[order], d[order]
    result = np.empty((k.size, z.size))
    if method == "failures":
        failed = t[d == 1]
        chunk = max(1, 262144 // failed.size)
        for start in range(0, z.size, chunk):
            distances = np.abs(z[start : start + chunk, None] - failed)
            result[:, start : start + chunk] = np.partition(distances, k - 1, axis=1)[:, k - 1].T
    else:
        unique, starts = np.unique(t, return_index=True)
        deaths = np.add.reduceat(d, starts)
        survival = np.cumprod(1 - deaths / (t.size - starts))
        if legacy and starts[-1] == t.size - 1:
            unique, survival = unique[:-1], survival[:-1]
        if unique.size == 0:
            raise ValueError("Archived Kaplan-Meier table is empty for this sample")
        for row, neighbors_i in enumerate(k):
            result[row] = _survival_radius(unique, survival, z, int(neighbors_i), t.size)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise RuntimeError("Nearest-neighbor rule produced a negative or nonfinite bandwidth")
    for array in (z, k, result):
        array.flags.writeable = False
    return NeighborBandwidths(z, k, result, method, bool(legacy))


def _survival_radius(
    unique: FloatArray, survival: FloatArray, grid: FloatArray, k: int, n: int
) -> FloatArray:
    """Batch ONEOLF distance trials and right-continuous survival lookups."""
    result = np.empty(grid.size)
    width = min(2 * k + 1, unique.size)
    offsets = np.arange(width)
    chunk = max(1, 262144 // width)
    threshold = 1.00001 * (k - 1) / n

    def at(points: FloatArray) -> FloatArray:
        index = np.searchsorted(unique, points, side="right") - 1
        return np.where(index < 0, 1, survival[np.maximum(index, 0)])

    def mass(z: FloatArray, radius: FloatArray) -> FloatArray:
        return at(z - radius) - at(z + radius)

    for start in range(0, grid.size, chunk):
        z = grid[start : start + chunk, None]
        pos = np.searchsorted(unique, z[:, 0], side="right")
        lo = np.maximum(pos - k - 1, 0)
        hi = np.minimum(pos + k, unique.size)
        indices = lo[:, None] + offsets
        valid = indices < hi[:, None]
        indices = np.minimum(indices, unique.size - 1)
        distances = np.sort(np.where(valid, np.abs(unique[indices] - z), np.inf), axis=1)
        lengths = hi - lo
        # Mask padded distances before arithmetic and table lookup.
        radii = np.where(np.isfinite(distances), distances, 0)
        exceed = (mass(z, radii) > threshold) & (offsets < lengths[:, None])
        has_exceed = exceed.any(axis=1)
        stop = np.where(has_exceed, exceed.argmax(axis=1), lengths - 1)
        rows = np.arange(z.size)
        r = radii[rows, stop]
        previous = np.where(has_exceed, stop - 1, stop)
        bw = np.where(previous >= 0, radii[rows, np.maximum(previous, 0)], -99.99)
        bigger = 1.00001 * bw
        smaller = 0.99999 * r
        selected = np.where(
            mass(z[:, 0], bigger) > threshold,
            bw,
            np.where(mass(z[:, 0], smaller) > threshold, bigger, smaller),
        )
        result[start : start + chunk] = selected
    return result
