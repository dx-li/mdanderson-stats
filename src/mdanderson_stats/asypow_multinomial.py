"""Multinomial information including the implicit final category."""

from math import fsum

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .boin import _owned


def asypow_multinomial_information(
    probabilities: ArrayLike, *, group_size: ArrayLike = 1
) -> FloatArray:
    """Information for K-1 free category probabilities; the last is 1-sum(p).

    Rows denote independent groups. All category masses and group sizes must
    be positive. Include the final category's rank-one contribution, omitted
    by original ASYPOW 2.1 when K>2. Output is per observation across groups.
    """
    p = finite(probabilities, "probabilities")
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or not 1 <= p.size <= 500 or not p.shape[1]:
        raise ValueError("probabilities need nonempty group rows and at most 500 parameters")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("category probabilities must be strictly between zero and one")
    remainder = np.array([fsum([1.0, *(-row)]) for row in p])
    if np.any(remainder <= 0):
        raise ValueError("each row must sum to less than one")
    weights = np.broadcast_to(finite(group_size, "group_size"), (len(p),))
    if np.any(weights <= 0):
        raise ValueError("group_size must be positive")
    log_weight = np.log(weights) - logsumexp(np.log(weights))
    result = np.zeros((p.size, p.size))
    size = p.shape[1]
    with np.errstate(over="ignore", under="ignore"):
        diagonal = np.exp(log_weight[:, None] - np.log(p))
        last = np.exp(log_weight - np.log(remainder))
        for group in range(len(p)):
            result[group * size : (group + 1) * size, group * size : (group + 1) * size] = (
                np.diag(diagonal[group]) + last[group]
            )
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("multinomial information exceeds floating-point range")
    return _owned(result)
