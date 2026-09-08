"""Fixed-bandwidth MUHAZ kernel hazard estimates under right censoring."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite, scalar

Kernel = Literal["rectangle", "epanechnikov", "biquadratic", "triquadratic"]
Boundary = Literal["none", "left", "both"]
_KERNELS = ("rectangle", "epanechnikov", "biquadratic", "triquadratic")
_BOUNDARIES = ("none", "left", "both")


@dataclass(frozen=True)
class MuhazFixed:
    time: FloatArray
    hazard: FloatArray
    bandwidth: float
    bounds: tuple[float, float]
    kernel: Kernel
    boundary: Boundary
    legacy: bool


def muhaz_fixed(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    bandwidth: float,
    grid: ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    kernel: Kernel = "epanechnikov",
    boundary: Boundary = "both",
    legacy: bool = False,
) -> MuhazFixed:
    """Smooth Nelson hazard increments with Mueller-Wang boundary kernels.

    The default groups tied failures into d/r increments and uses closed kernel
    support. legacy=True reproduces archived HAZDEN's sequential risk weights,
    stable input order for ties and asymmetric support endpoints. Negative
    boundary-kernel hazard totals are truncated at zero, as in MUHAZ.

    Default bounds are (0, max(times)); default grid is 101 equally spaced points.
    This explicit-bandwidth core does not select bandwidths or the ten-at-risk
    upper bound of the archived high-level muhaz function.
    """
    t = finite(times, "times")
    if t.ndim != 1 or t.size == 0 or np.any(t < 0):
        raise ValueError("times must be a nonempty nonnegative vector")
    status = np.ones(t.size) if delta is None else count(delta, "delta")
    if status.shape != t.shape or np.any(status > 1):
        raise ValueError("delta must match times and contain only 0 or 1")
    b = scalar(bandwidth, "bandwidth")
    if b <= 0:
        raise ValueError("bandwidth must be positive")
    if kernel not in _KERNELS or boundary not in _BOUNDARIES:
        raise ValueError("Unknown kernel or boundary correction")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    interval = finite((0, float(t.max())) if bounds is None else bounds, "bounds")
    if interval.shape != (2,) or not 0 <= interval[0] < interval[1]:
        raise ValueError("bounds must be two increasing nonnegative values")
    left, right = map(float, interval)
    z = np.linspace(left, right, 101) if grid is None else finite(grid, "grid").copy()
    if z.ndim != 1 or np.any((z < left) | (z > right)):
        raise ValueError("grid must be a vector within bounds")
    order = np.argsort(t, kind="stable")
    t, status = t[order], status[order]
    if legacy:
        weights = status / np.arange(t.size, 0, -1)
        event_time, weights = t[status == 1], weights[status == 1]
    else:
        unique, starts = np.unique(t, return_index=True)
        deaths = np.add.reduceat(status, starts)
        event_time = unique[deaths > 0]
        weights = (deaths / (t.size - starts))[deaths > 0]
    hazard = np.zeros(z.size)
    # Bound intermediate storage while vectorizing both grid and event axes.
    chunk = max(1, 262144 // max(1, event_time.size))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, z.size, chunk):
                points = z[start : start + chunk, None]
                u = (points - event_time) / b
                if legacy:
                    support = (event_time > points - b) & (
                        (event_time < points + b) | (points + b >= t[-1])
                    )
                else:
                    support = np.abs(u) <= 1
                q = np.ones_like(points)
                left_edge = (points < left + b) & (boundary != "none")
                right_edge = (~left_edge) & (points > right - b) & (boundary == "both")
                q = np.where(left_edge, (points - left) / b, q)
                q = np.where(right_edge, (right - points) / b, q)
                u = np.where(right_edge, -u, u)
                support &= (~right_edge) | (u <= q)
                if not legacy:
                    support &= (~left_edge) | (u <= q)
                # Values outside support do not contribute and must not overflow.
                values = _kernel(np.where(support, u, 0), q, _KERNELS.index(kernel))
                hazard[start : start + chunk] = np.maximum(
                    0, np.sum(np.where(support, values, 0) * weights, axis=1) / b
                )
    except FloatingPointError as exc:
        raise RuntimeError("MUHAZ kernel calculation exceeds numerical range") from exc
    z.flags.writeable = hazard.flags.writeable = False
    return MuhazFixed(z, hazard, b, (left, right), kernel, boundary, bool(legacy))


def _kernel(u: FloatArray, q: FloatArray, shape: int) -> FloatArray:
    """Polynomial boundary kernels (Mueller and Wang, Table 1)."""
    if shape == 0:
        interior = np.full_like(u, 0.5)
        edge = 2 * (2 * (1 - q + q * q) + 3 * (1 - q) * u) / (1 + q) ** 3
    elif shape == 1:
        interior = 0.75 * (1 - u * u)
        edge = 12 * (u + 1) * (0.5 * (3 * q * q - 2 * q + 1) + u * (1 - 2 * q)) / (1 + q) ** 4
    elif shape == 2:
        interior = 15 * (1 - u * u) ** 2 / 16
        edge = (
            60 * (u + 1) ** 2 * (q - u) * (2 * q * q - 2 * q + 1 + u * (2 - 3 * q)) / (1 + q) ** 6
        )
    else:
        interior = 35 * (1 - u * u) ** 3 / 32
        edge = (
            280
            * (1 + u) ** 3
            * (q - u) ** 2
            * (0.5 * (5 * q * q - 6 * q + 3) + u * (3 - 4 * q))
            / (1 + q) ** 8
        )
    return np.where(q == 1, interior, edge)
