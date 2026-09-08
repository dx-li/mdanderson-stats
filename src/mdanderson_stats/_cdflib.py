"""Shared immutable results and complementary-pair validation for CDFLIB ports."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _pair(
    first: ArrayLike | None, second: ArrayLike | None, name: str
) -> tuple[FloatArray, FloatArray]:
    if first is None and second is None:
        raise ValueError(f"provide at least one of {name}")
    supplied = first if first is not None else second
    assert supplied is not None
    p = finite(supplied, name)
    q = 1 - p if first is None or second is None else finite(second, name)
    if first is None:
        p, q = q, p
    p, q = np.broadcast_arrays(p, q)
    if np.any((p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ValueError(f"{name} must lie in [0,1]")
    if np.any(np.abs((p + q) - 1) > 8 * np.finfo(float).eps):
        raise ValueError(f"{name} must sum to one within 8 machine epsilons")
    # Preserve the smaller supplied number; reconstruct only the larger one.
    return np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)
