"""Pointwise MUHAZ bandwidth selection and boundary-kernel smoothing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, scalar
from .muhaz import _KERNELS, Boundary, Kernel, _event_weights, _hazard_values, _kernel, muhaz_fixed
from .muhaz_global import _prepare_selection
from .muhaz_mse import MuhazMSE, muhaz_mse


@dataclass(frozen=True)
class MuhazLocal:
    time: FloatArray
    hazard: FloatArray
    bandwidth: FloatArray
    local_bandwidth: FloatArray | None
    selected_index: NDArray[np.int64] | None
    minimum_mse: FloatArray | None
    selected_bias: FloatArray | None
    selected_variance: FloatArray | None
    score: float | None
    diagnostics: MuhazMSE | None
    smoothing_bandwidth: float | None
    bounds: tuple[float, float]
    kernel: Kernel
    boundary: Boundary
    legacy: bool
    n_observations: int
    n_events: int
    pilot_bandwidth: float | None = None
    n_min_grid: int | None = None


def muhaz_local(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    bandwidths: ArrayLike | None = None,
    pilot_bandwidth: float | None = None,
    smoothing_bandwidth: float | None = None,
    bounds: tuple[float | None, float | None] | None = None,
    subset: ArrayLike | None = None,
    n_min_grid: int = 51,
    n_est_grid: int = 101,
    kernel: Kernel = "epanechnikov",
    boundary: Boundary = "both",
    legacy: bool = False,
    rtol: float = 0.001,
    max_refinements: int = 6,
) -> MuhazLocal:
    """Select a bandwidth at each MSE grid point, smooth it, and fit the hazard.

    Default settings and subsetting match muhaz_global. Smoothing bandwidth
    defaults to five times the pilot. A single candidate bypasses selection and
    smoothing. Undefined or nonpositive smoothed bandwidths raise RuntimeError.
    Legacy mode preserves source selection/smoothing quirks; unavailable source
    bias/variance diagnostics are represented by NaN rather than memory contents.
    """
    t, d, bw, pilot, interval = _prepare_selection(
        times,
        delta,
        bandwidths,
        pilot_bandwidth,
        bounds,
        subset,
        n_min_grid,
        n_est_grid,
        legacy,
    )
    smooth = (
        None if smoothing_bandwidth is None else scalar(smoothing_bandwidth, "smoothing_bandwidth")
    )
    if smooth is not None and smooth <= 0:
        raise ValueError("smoothing_bandwidth must be positive")
    z = np.linspace(*interval, n_est_grid)
    diagnostic = None
    selected = local_bw = minimum = bias = variance = None
    score = None
    if bw.size == 1:
        fixed = muhaz_fixed(
            t,
            d,
            bandwidth=float(bw[0]),
            grid=z,
            bounds=interval,
            kernel=kernel,
            boundary=boundary,
            legacy=legacy,
        )
        hazard = fixed.hazard
        used_bw = np.full(z.size, bw[0])
        smooth = None
    else:
        assert pilot is not None
        if smooth is None:
            smooth = 5 * pilot
        if not np.isfinite(smooth):
            raise ValueError("Default smoothing bandwidth exceeds numerical range")
        diagnostic = muhaz_mse(
            t,
            d,
            bandwidths=bw,
            pilot_bandwidth=pilot,
            grid=np.linspace(*interval, n_min_grid),
            bounds=interval,
            kernel=kernel,
            boundary=boundary,
            legacy=legacy,
            rtol=rtol,
            max_refinements=max_refinements,
        )
        mse = diagnostic.mse
        available = np.ones(mse.shape[1], dtype=bool)
        if legacy:
            eligible = (mse > 0) & (mse < 1e30)
            available = eligible.any(axis=0)
            selected = np.where(
                available, np.argmin(np.where(eligible, mse, np.inf), axis=0), bw.size - 1
            )
        else:
            selected = np.argmin(mse, axis=0)
        columns = np.arange(mse.shape[1])
        minimum = np.where(available, mse[selected, columns], 1e30)
        bias = np.where(available, diagnostic.bias[selected, columns], np.nan)
        variance = np.where(available, diagnostic.variance[selected, columns], np.nan)
        score = float(minimum.sum())
        if not np.isfinite(score):
            raise RuntimeError("Summed local MSE exceeds numerical range")
        local_bw = bw[selected]
        used_bw = _smooth_bandwidth(
            diagnostic.time, local_bw, z, smooth, interval, kernel, boundary, legacy
        )
        order = np.argsort(t, kind="stable")
        t, d = t[order], d[order]
        event_time, weights = _event_weights(t, d, bool(legacy))
        hazard = _hazard_values(
            event_time, weights, t[-1], z, used_bw, *interval, kernel, boundary, bool(legacy)
        )
    for array in (z, hazard, used_bw, local_bw, selected, minimum, bias, variance):
        if array is not None:
            array.flags.writeable = False
    return MuhazLocal(
        z,
        hazard,
        used_bw,
        local_bw,
        selected,
        minimum,
        bias,
        variance,
        score,
        diagnostic,
        smooth,
        interval,
        kernel,
        boundary,
        bool(legacy),
        t.size,
        int(d.sum()),
        pilot_bandwidth=pilot,
        n_min_grid=int(n_min_grid),
    )


def _smooth_bandwidth(
    grid: FloatArray,
    bandwidth: FloatArray,
    points: FloatArray,
    smoothing: float,
    bounds: tuple[float, float],
    kernel: Kernel,
    boundary: Boundary,
    legacy: bool,
) -> FloatArray:
    """Kernel regression with bounded grid-by-estimation-point intermediates."""
    left, right = bounds
    result = np.empty(points.size)
    chunk = max(1, 262144 // grid.size)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, points.size, chunk):
                z = points[start : start + chunk, None]
                u = (z - grid) / smoothing
                support = np.abs(u) <= 1
                if legacy:
                    support = (grid > z - smoothing) & (
                        (grid < z + smoothing) | (z + smoothing >= grid[-1])
                    )
                left_edge = (z < left + smoothing) & (boundary != "none")
                right_edge = (
                    (~left_edge)
                    & (z > right - smoothing)
                    & ((boundary == "both") | (legacy & (boundary == "left")))
                )
                q = np.where(
                    left_edge,
                    (z - left) / smoothing,
                    np.where(right_edge, (right - z) / smoothing, 1),
                )
                u = np.where(right_edge, -u, u)
                values = np.where(
                    support, _kernel(np.where(support, u, 0), q, _KERNELS.index(kernel)), 0
                )
                result[start : start + chunk] = np.sum(values * bandwidth, axis=1) / np.sum(
                    values, axis=1
                )
    except FloatingPointError as exc:
        raise RuntimeError(
            "Local bandwidth smoothing is undefined; revise the smoothing bandwidth or grids"
        ) from exc
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise RuntimeError(
            "Smoothed bandwidth must be finite and positive; "
            "revise the smoothing bandwidth or grids"
        )
    return result
