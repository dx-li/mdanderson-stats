"""Partition constraints for continuous SINGLE allocations."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


def allocation_constraints(
    lengths: list[int], group_totals: ArrayLike | None, total: float
) -> tuple[FloatArray, FloatArray]:
    if group_totals is None:
        return np.ones((1, sum(lengths))), np.ones(1)
    totals = finite(group_totals, "group_totals")
    if len(lengths) != 2 or totals.shape != (2,) or np.any(totals <= 0):
        raise ValueError("group_totals requires two positive totals for a two-sample design")
    if not np.isclose(totals.sum(), total, rtol=1e-12, atol=0):
        raise ValueError("group_totals must sum to total_subjects")
    matrix = np.zeros((2, sum(lengths)))
    matrix[0, : lengths[0]] = 1
    matrix[1, lengths[0] :] = 1
    return matrix, totals / totals.sum()


def allocation_gap(
    fractions: FloatArray,
    derivative: FloatArray,
    matrix: FloatArray,
    scale: float,
) -> float:
    sensitivity = -derivative
    means = (matrix @ (fractions * sensitivity)) / (matrix @ fractions)
    return float(max(0.0, np.max(sensitivity - matrix.T @ means) / scale))
