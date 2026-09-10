"""Exact expected-likelihood projection under fixed categorical parameters."""

from math import fsum

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .asypow_constraints import _components


def _fixed_categorical_null(
    p: FloatArray, mass: FloatArray, constraints: ArrayLike, ordinal: bool
) -> tuple[FloatArray, int]:
    fixed = np.full(p.size, np.nan)
    for indices, value in _components(constraints, p.size):
        if value is not None:
            if not 0 < value < 1:
                raise ValueError("fixed categorical parameters must be strictly within (0,1)")
            fixed[indices] = value
        elif len(indices) > 1:
            raise NotImplementedError(
                "unfixed categorical equality components are not yet supported"
            )
    df = int(np.count_nonzero(~np.isnan(fixed)))
    if df == 0:
        raise ValueError("constraints must impose at least one independent restriction")
    fixed_rows = fixed.reshape(p.shape)
    q = p.copy()
    for g, row in enumerate(p):
        selected = np.flatnonzero(~np.isnan(fixed_rows[g]))
        if not len(selected) or np.array_equal(row[selected], fixed_rows[g, selected]):
            continue
        q[g, selected] = fixed_rows[g, selected]
        if ordinal:
            # Fixed cumulative thresholds split the categories into segments.
            # Preserve the alternative's conditional probabilities in each one.
            anchors = np.r_[-1, selected, len(row)]
            original = np.r_[0, row[selected], 1]
            target = np.r_[0, fixed_rows[g, selected], 1]
            if np.any(np.diff(target) <= 0):
                raise ValueError("fixed cumulative probabilities must increase strictly")
            for j, (left, right) in enumerate(zip(anchors[:-1], anchors[1:], strict=True)):
                interior = slice(left + 1, right)
                q[g, interior] = target[j] + (target[j + 1] - target[j]) * (
                    (row[interior] - original[j]) / (original[j + 1] - original[j])
                )
        else:
            remaining = fsum([1.0, *(-fixed_rows[g, selected])])
            if remaining <= 0:
                raise ValueError("fixed category probabilities must sum to less than one")
            free = np.isnan(fixed_rows[g])
            original_remaining = fsum([*row[free], mass[g, -1]])
            q[g, free] = remaining * (row[free] / original_remaining)
    return q, df
