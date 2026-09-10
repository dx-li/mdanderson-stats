"""Nearest-neighbor membership and boundary weights for WINDOWS."""

import numpy as np

from ._validation import FloatArray


def _neighbor_bounds(x: FloatArray, center: float, k: int) -> tuple[int, int]:
    # Binary-search the leftmost closest block, then include distance-boundary ties.
    lo, hi = 0, x.size - k
    while lo < hi:
        mid = (lo + hi) // 2
        if center / 2 - x[mid] / 2 > x[mid + k] / 2 - center / 2:
            lo = mid + 1
        else:
            hi = mid
    with np.errstate(over="ignore"):
        radius = max(abs(center - x[lo]), abs(x[lo + k - 1] - center))
    if not np.isfinite(radius):
        raise ArithmeticError("neighbor distances are not representable; rescale x")
    with np.errstate(over="ignore"):
        lower, upper = center - radius, center + radius
    # Include the chosen block even if endpoint reconstruction rounds inward.
    left = min(lo, int(np.searchsorted(x, lower, side="left")))
    right = max(lo + k, int(np.searchsorted(x, upper, side="right")))
    return left, right


def _neighbor_weights(x: FloatArray, center: float, k: int, weighting: str) -> FloatArray:
    n = x.size
    low = x == x[0]
    high = x == x[-1]
    boundary = low | high
    count = int(boundary.sum())
    factor = (k - (n - count)) / count
    if not 0 < factor <= 1:
        raise ArithmeticError("neighbor boundary mass is numerically inconsistent")
    if weighting == "boxcar":
        return np.where(boundary, factor, 1.0)
    with np.errstate(over="ignore"):
        span = x[-1] - x[0]
    if not np.isfinite(span):
        raise ArithmeticError("neighbor span is not representable; rescale x")
    if span == 0:
        if center == x[0]:
            return np.ones(n)
        return np.zeros(n)
    # Log weights also support centers far outside a narrowly spaced data range.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_z = np.log(np.abs(x - center)) - np.log(span)
        log_weight = np.where(
            log_z > 0,
            4 * log_z + 2 * np.log(-np.expm1(-2 * log_z)),
            2 * np.log(-np.expm1(2 * log_z)),
        )
    if count > 2:
        log_weight[boundary] += np.log(factor)
    maximum = np.max(log_weight)
    if not np.isfinite(maximum):
        return np.zeros(n)
    return np.exp(log_weight - maximum)
