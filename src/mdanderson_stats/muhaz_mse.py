"""MUHAZ pilot-based bias/variance and candidate bandwidth diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .muhaz import _KERNELS, Boundary, Kernel, _event_weights, _hazard_values, _kernel, muhaz_fixed


@dataclass(frozen=True)
class MuhazMSE:
    time: FloatArray
    bandwidths: FloatArray
    bias: FloatArray
    variance: FloatArray
    mse: FloatArray
    pilot_hazard: FloatArray
    refinements: NDArray[np.int64]
    converged: NDArray[np.bool_]
    pilot_bandwidth: float
    bounds: tuple[float, float]
    kernel: Kernel
    boundary: Boundary
    legacy: bool
    rtol: float
    max_refinements: int


def muhaz_mse(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    bandwidths: ArrayLike,
    pilot_bandwidth: float,
    grid: ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    kernel: Kernel = "epanechnikov",
    boundary: Boundary = "both",
    legacy: bool = False,
    rtol: float = 0.001,
    max_refinements: int = 6,
) -> MuhazMSE:
    """Pilot-convolution MSE over candidate bandwidths (rows) and times (columns).

    A scalar/vector gives constant candidates; a matrix supplies a varying
    bandwidth for every candidate and grid point.

    Uses the archived survival factor 1 - events_at_or_before/(N+1), which is
    neither Kaplan-Meier survival nor a survival estimate conditional on failure.
    Trapezoid refinement stops when both integrals meet relative tolerance, or
    at max_refinements. Convergence diagnostics distinguish those outcomes.
    No bandwidth is selected by this function.
    """
    candidates = finite(bandwidths, "bandwidths")
    if candidates.ndim > 2 or candidates.size == 0 or np.any(candidates <= 0):
        raise ValueError(
            "bandwidths must be positive: a scalar, vector, or candidate-by-time matrix"
        )
    candidates = candidates.reshape(-1).copy() if candidates.ndim < 2 else candidates.copy()
    tolerance = scalar(rtol, "rtol")
    if tolerance < 0:
        raise ValueError("rtol must be nonnegative")
    if (
        isinstance(max_refinements, (bool, np.bool_))
        or not isinstance(max_refinements, (int, np.integer))
        or not 1 <= max_refinements <= 16
    ):
        raise ValueError("max_refinements must be an integer from 1 to 16")
    # Reuse the fixed-estimator input contract and its pilot evaluation.
    pilot = muhaz_fixed(
        times,
        delta,
        bandwidth=pilot_bandwidth,
        grid=grid,
        bounds=bounds,
        kernel=kernel,
        boundary=boundary,
        legacy=legacy,
    )
    t = np.asarray(times, dtype=float)
    status = np.ones(t.size) if delta is None else np.asarray(delta, dtype=float)
    order = np.argsort(t, kind="stable")
    t, status = t[order], status[order]
    event_time, weights = _event_weights(t, status, bool(legacy))
    failed = t[status == 1]
    left, right = pilot.bounds
    if candidates.ndim == 2 and candidates.shape[1] != pilot.time.size:
        raise ValueError("Bandwidth matrix columns must match the evaluation grid")
    z, b = np.broadcast_arrays(
        pilot.time[None, :], candidates[:, None] if candidates.ndim == 1 else candidates
    )
    shape = z.shape
    z, b = z.ravel(), b.ravel()
    left_edge = (z < left + b) & (boundary != "none")
    right_edge = (~left_edge) & (z > right - b) & (boundary == "both")
    q = np.where(left_edge, (z - left) / b, np.where(right_edge, (right - z) / b, 1))
    lower = np.where(right_edge, -q, -1)
    upper = np.where(left_edge, q, 1)
    # FUNC reflects near the right boundary even if an overlapping left
    # correction was chosen. Compatibility mode retains that source behavior.
    reflect = (z > right - b) if legacy else right_edge
    integral_bias, integral_var = np.zeros(z.size), np.zeros(z.size)
    converged = np.zeros(z.size, dtype=bool)
    levels = np.zeros(z.size, dtype=np.int64)

    def integrand(
        indices: NDArray[np.int64], fraction: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        means_bias, means_var = np.empty(indices.size), np.empty(indices.size)
        selected = indices
        chunk = max(1, 262144 // fraction.size)
        for start in range(0, selected.size, chunk):
            stop = min(start + chunk, selected.size)
            indices = selected[start:stop]
            u = lower[indices, None] + (upper - lower)[indices, None] * fraction
            if legacy:
                if fraction[0] == 0:
                    u = np.column_stack((lower[indices], upper[indices]))
                else:
                    # TRY advances xx by repeated addition, which matters when
                    # rounding moves a pilot query across a discontinuity.
                    step = (upper - lower)[indices] / fraction.size
                    increments = np.broadcast_to(step[:, None], u.shape).copy()
                    increments[:, 0] = lower[indices] + 0.5 * step
                    u = np.cumsum(increments, axis=1)

            at = z[indices, None] - b[indices, None] * u
            h = _hazard_values(
                event_time,
                weights,
                t[-1],
                at,
                pilot.bandwidth,
                left,
                right,
                kernel,
                boundary,
                bool(legacy),
            )
            k = _kernel(
                np.where(reflect[indices, None], -u, u), q[indices, None], _KERNELS.index(kernel)
            )
            survival = 1 - np.searchsorted(failed, at, side="right") / (t.size + 1)
            means_bias[start:stop] = (h * k).mean(axis=1)
            means_var[start:stop] = (h * k * k / survival).mean(axis=1)
        return means_bias, means_var

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for level in range(1, max_refinements + 1):
                active = np.flatnonzero(~converged)
                if active.size == 0:
                    break
                if level == 1:
                    ib, iv = integrand(active, np.array([0.0, 1.0]))
                    new_bias = (upper - lower)[active] * ib
                    new_var = (upper - lower)[active] * iv
                else:
                    n = 2 ** (level - 2)
                    ib, iv = integrand(active, (np.arange(n) + 0.5) / n)
                    new_bias = (integral_bias[active] + (upper - lower)[active] * ib) / 2
                    new_var = (integral_var[active] + (upper - lower)[active] * iv) / 2
                    converged[active] = (
                        np.abs(new_bias - integral_bias[active])
                        <= tolerance * np.abs(integral_bias[active])
                    ) & (
                        np.abs(new_var - integral_var[active])
                        <= tolerance * np.abs(integral_var[active])
                    )
                integral_bias[active], integral_var[active] = new_bias, new_var
                levels[active] = level
            bias = integral_bias.reshape(shape) - pilot.hazard
            variance = (integral_var / (t.size * b)).reshape(shape)
            mse = bias * bias + variance
    except FloatingPointError as exc:
        raise RuntimeError("MUHAZ MSE calculation exceeds numerical range") from exc
    levels_grid, converged_grid = levels.reshape(shape), converged.reshape(shape)
    for array in [candidates, bias, variance, mse, levels_grid, converged_grid]:
        array.flags.writeable = False
    return MuhazMSE(
        pilot.time,
        candidates,
        bias,
        variance,
        mse,
        pilot.hazard,
        levels_grid,
        converged_grid,
        pilot.bandwidth,
        pilot.bounds,
        kernel,
        boundary,
        bool(legacy),
        tolerance,
        int(max_refinements),
    )
