"""Maximize nonnegative Bernstein-form rejection probabilities."""

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from ._validation import FloatArray


@dataclass(frozen=True)
class ExtsigMaximum:
    pvalue: float
    lower_bound: float
    nuisance_probability: float
    error_bound: float


def _split(coefficients: FloatArray) -> tuple[FloatArray, FloatArray]:
    work = coefficients.copy()
    n = work.size - 1
    left, right = np.empty(n + 1), np.empty(n + 1)
    left[0], right[n] = work[0], work[n]
    for i in range(1, n + 1):
        work = (work[:-1] + work[1:]) / 2
        left[i], right[n - i] = work[0], work[-1]
    return left, right


def _maximum(coefficients: FloatArray, method: str, tolerance: float) -> ExtsigMaximum:
    n = coefficients.size - 1
    scale = float(coefficients.max())
    if scale == 0:
        return ExtsigMaximum(0.0, 0.0, 0.01 if method == "native_grid" else 0.0, 0.0)
    coefficients = coefficients / scale
    grid = np.arange(1, 100) / 100 if method == "native_grid" else np.linspace(0, 1, 101)
    values = coefficients @ binom.pmf(np.arange(n + 1)[:, None], n, grid)
    best_index = int(np.argmax(values))
    best, at = float(values[best_index]), float(grid[best_index])
    if best * scale == 0:
        raise ArithmeticError("EXTSIG rejection probability underflows float64")
    if method == "native_grid":
        value = min(1.0, best * scale)
        return ExtsigMaximum(value, value, at, 0.0)
    # Convex-hull bounds remain valid after subdivision; inspect every region
    # whose upper bound can improve the global maximum, not only grid peaks.
    serial = 0
    heap = [(-float(coefficients.max()), serial, 0.0, 1.0, coefficients)]
    while heap and -heap[0][0] > best + tolerance:
        _, _, lo, hi, c = heapq.heappop(heap)
        midpoint = (lo + hi) / 2
        left, right = _split(c)
        if left[-1] > best:
            best, at = float(left[-1]), midpoint
        for a, b, sub in [(lo, midpoint, left), (midpoint, hi, right)]:
            upper = float(sub.max())
            if upper > best:
                serial += 1
                heapq.heappush(heap, (-upper, serial, a, b, sub))
        if serial > 20_000:
            raise ArithmeticError("EXTSIG nuisance bound did not converge within subdivision limit")
    upper = max(best, -heap[0][0] if heap else best)
    # Allow for floating-point evaluation of conditional probabilities and
    # subdivision. This is a numerical bound, not interval-arithmetic proof.
    padding = 32 * np.finfo(float).eps * (n + 1)
    upper = min(1.0, upper + padding)
    lower = max(0.0, best - padding)
    return ExtsigMaximum(
        float(min(1.0, upper * scale)), float(lower * scale), at, float((upper - lower) * scale)
    )
