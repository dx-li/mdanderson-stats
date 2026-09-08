"""MUHAZ nearest-neighbor selection, bandwidth smoothing and hazard fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, scalar
from .muhaz import Boundary, Kernel, _event_weights, _hazard_values, muhaz_fixed
from .muhaz_global import _prepare_selection
from .muhaz_local import _smooth_bandwidth
from .muhaz_mse import MuhazMSE, muhaz_mse
from .muhaz_neighbors import NeighborBandwidths, NeighborMethod, muhaz_neighbor_bandwidths


@dataclass(frozen=True)
class MuhazKNN:
    time: FloatArray
    hazard: FloatArray
    bandwidth: FloatArray
    neighbor_bandwidths: NeighborBandwidths
    selected_index: int
    neighbors: int
    scores: FloatArray | None
    score: float | None
    diagnostics: MuhazMSE | None
    smoothing_bandwidth: float
    bounds: tuple[float, float]
    kernel: Kernel
    boundary: Boundary
    legacy: bool
    n_observations: int
    n_events: int


def muhaz_knn(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    neighbors: ArrayLike | None = None,
    method: NeighborMethod = "survival",
    pilot_bandwidth: float | None = None,
    smoothing_bandwidth: float | None = None,
    bounds: tuple[float, float] | None = None,
    subset: ArrayLike | None = None,
    n_min_grid: int = 51,
    n_est_grid: int = 101,
    kernel: Kernel = "epanechnikov",
    boundary: Boundary = "both",
    legacy: bool = False,
    rtol: float = 0.001,
    max_refinements: int = 6,
) -> MuhazKNN:
    """Choose one neighbor count by summed grid MSE, smooth radii and fit hazard.

    Defaults to counts 2 through floor(events/2), and the survival-mass method
    selected by the archived S interface. Single-count input bypasses MSE but
    still smooths its bandwidths. Default time/pilot conventions match muhaz_global.
    """
    t, d, _, pilot, interval = _prepare_selection(
        times,
        delta,
        None,
        pilot_bandwidth,
        bounds,
        subset,
        n_min_grid,
        n_est_grid,
        legacy,
    )
    assert pilot is not None
    if neighbors is None:
        upper = int(d.sum()) // 2
        if upper < 2:
            raise ValueError("Default neighbor range requires at least four observed failures")
        neighbors = np.arange(2, upper + 1)
    radii = muhaz_neighbor_bandwidths(
        t,
        d,
        neighbors=neighbors,
        grid=np.linspace(*interval, n_min_grid),
        method=method,
        legacy=legacy,
    )
    smooth = (
        5 * pilot
        if smoothing_bandwidth is None
        else scalar(smoothing_bandwidth, "smoothing_bandwidth")
    )
    if not np.isfinite(smooth) or smooth <= 0:
        raise ValueError("smoothing_bandwidth must be finite and positive")
    selected = 0
    diagnostic = None
    scores = None
    score = None
    if radii.neighbors.size > 1:
        if np.any(radii.bandwidth <= 0):
            raise ValueError(
                "Zero neighbor radii cannot be used in MSE; revise neighbor counts or grid"
            )
        diagnostic = muhaz_mse(
            t,
            d,
            bandwidths=radii.bandwidth,
            pilot_bandwidth=pilot,
            grid=radii.time,
            bounds=interval,
            kernel=kernel,
            boundary=boundary,
            legacy=legacy,
            rtol=rtol,
            max_refinements=max_refinements,
        )
        scores = diagnostic.mse.sum(axis=1)
        if np.any(~np.isfinite(scores)):
            raise RuntimeError("Summed nearest-neighbor MSE exceeds numerical range")
        if legacy and not np.any(scores < 1e5):
            raise RuntimeError(
                "Archived neighbor selector has no defined optimum below its 1e5 cutoff"
            )
        selected = int(np.argmin(scores))
        score = float(scores[selected])
        scores.flags.writeable = False
    else:
        # Validate the hazard settings even when no MSE/pilot evaluation is needed.
        muhaz_fixed(
            t,
            d,
            bandwidth=pilot,
            grid=[],
            bounds=interval,
            kernel=kernel,
            boundary=boundary,
            legacy=legacy,
        )
    z = np.linspace(*interval, n_est_grid)
    used_bw = _smooth_bandwidth(
        radii.time, radii.bandwidth[selected], z, smooth, interval, kernel, boundary, legacy
    )
    order = np.argsort(t, kind="stable")
    t, d = t[order], d[order]
    event_time, weights = _event_weights(t, d, bool(legacy))
    hazard = _hazard_values(
        event_time, weights, t[-1], z, used_bw, *interval, kernel, boundary, bool(legacy)
    )
    for array in (z, used_bw, hazard):
        array.flags.writeable = False
    return MuhazKNN(
        z,
        hazard,
        used_bw,
        radii,
        selected,
        int(radii.neighbors[selected]),
        scores,
        score,
        diagnostic,
        smooth,
        interval,
        kernel,
        boundary,
        bool(legacy),
        t.size,
        int(d.sum()),
    )
