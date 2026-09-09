"""Shared vectorized search for legacy distribution df inversions."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ._validation import FloatArray


def _invert_df(
    target: FloatArray,
    low: FloatArray,
    high: FloatArray,
    evaluate: Callable[[FloatArray, NDArray[np.intp]], FloatArray],
    *,
    probability_atol: FloatArray | float = 0.0,
) -> FloatArray:
    """Find a crossing; evaluate receives values and original flattened row indices.

    Inputs have already been broadcast and validated by the distribution.
    Indices keep fixed parameters aligned when endpoint solutions are removed.
    The caller verifies the final probability against the requested tail.
    """
    probability_atol = np.broadcast_to(probability_atol, target.shape).ravel()
    shape = target.shape
    target, low, high = target.ravel(), low.ravel(), high.ravel()
    indices = np.arange(target.size, dtype=np.intp)
    at_low, at_high = evaluate(low, indices), evaluate(high, indices)
    if np.any(at_low == at_high):
        raise ValueError("df is numerically unidentified across this bracket")
    adjusted = target.copy()
    for endpoint in (at_low, at_high):
        adjusted = np.where(
            np.abs(target - endpoint) <= 32 * np.finfo(float).eps * endpoint + probability_atol,
            endpoint,
            adjusted,
        )
    if np.any((adjusted < np.minimum(at_low, at_high)) | (adjusted > np.maximum(at_low, at_high))):
        raise ValueError("df root is not bracketed; multiple roots may require df_bracket")
    resolved = (adjusted == at_low) | (adjusted == at_high)
    answer = np.where(adjusted == at_low, low, high)
    if np.all(resolved):
        return answer.reshape(shape)
    active = ~resolved
    indices, adjusted = indices[active], adjusted[active]
    low, high, at_low = low[active], high[active], at_low[active]
    # Establish a local crossing from the legacy initial value of five.
    # Huge log-midpoints can reach ill-conditioned beta parameters even
    # when the desired df is ordinary and nearby.
    current = np.clip(np.full(low.shape, 5.0), low, high)
    current_residual = evaluate(current, indices) - adjusted
    left = np.signbit(current_residual) != np.signbit(at_low - adjusted)
    found = current_residual == 0
    bracket_low, bracket_high = current.copy(), current.copy()
    for _ in range(300):
        if np.all(found):
            break
        trial = np.where(left, np.maximum(low, current / 5), np.minimum(high, current * 5))
        trial = np.where(found, current, trial)
        residual = evaluate(trial, indices) - adjusted
        crossed = (np.signbit(residual) != np.signbit(current_residual)) | (residual == 0)
        newly_found = ~found & crossed
        bracket_low = np.where(newly_found, np.minimum(current, trial), bracket_low)
        bracket_high = np.where(newly_found, np.maximum(current, trial), bracket_high)
        found |= crossed
        current, current_residual = trial, residual
    if not np.all(found):
        raise ArithmeticError("legacy df search could not isolate a crossing")
    lo, hi = np.log(bracket_low), np.log(bracket_high)
    residual_low = evaluate(bracket_low, indices) - adjusted
    best, error = bracket_low.copy(), np.abs(residual_low)
    for _ in range(64):
        middle = (lo + hi) / 2
        candidate = np.clip(np.exp(middle), bracket_low, bracket_high)
        residual = evaluate(candidate, indices) - adjusted
        best = np.where(np.abs(residual) < error, candidate, best)
        error = np.minimum(error, np.abs(residual))
        move_low = np.signbit(residual) == np.signbit(residual_low)
        lo, hi = np.where(move_low, middle, lo), np.where(move_low, hi, middle)
        residual_low = np.where(move_low, residual, residual_low)
    answer[active] = best
    return answer.reshape(shape)
