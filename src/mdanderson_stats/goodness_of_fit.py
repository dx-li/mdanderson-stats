"""Pearson goodness-of-fit tests corresponding to GOFCHI."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc

from ._validation import FloatArray, count, finite


@dataclass(frozen=True)
class GoodnessOfFit:
    statistic: FloatArray
    pvalue: FloatArray
    degrees_of_freedom: FloatArray
    expected: FloatArray
    contributions: FloatArray


def chi_square_gof(
    observed: ArrayLike,
    weights: ArrayLike,
    *,
    degrees_of_freedom: ArrayLike | None = None,
    legacy_zero_expected: bool = False,
) -> GoodnessOfFit:
    """Normalize null weights and test observed frequencies along the last axis.

    Leading dimensions broadcast into batches of independent tests. Frequencies
    may be fractional, as in GOFCHI. Supply degrees_of_freedom when parameters
    were estimated; otherwise use the number of positive null weights minus one.

    A positive observation in a category with zero null weight gives an infinite
    statistic and p=0. ``legacy_zero_expected=True`` reproduces GOFCHI's behavior
    of ignoring such contributions, which is generally not a valid test.
    Zero-observed/zero-expected categories contribute zero in either mode.
    """
    observed, weights = np.broadcast_arrays(
        finite(observed, "observed"), finite(weights, "weights")
    )
    if observed.ndim < 1 or observed.shape[-1] < 2:
        raise ValueError("At least two categories are required along the last axis")
    if np.any(observed < 0) or np.any(weights < 0):
        raise ValueError("observed and weights must be nonnegative")
    total = observed.sum(axis=-1, keepdims=True)
    # Scaling the weights before summing avoids overflow from arbitrary units.
    scale = weights.max(axis=-1, keepdims=True)
    if np.any(scale == 0) or np.any(total == 0) or not np.all(np.isfinite(total)):
        raise ValueError(
            "Observed totals must be positive and finite; weights must have positive sum"
        )
    normalized = weights / scale
    expected = normalized / normalized.sum(axis=-1, keepdims=True) * total
    df_values = (
        (weights > 0).sum(axis=-1) - 1
        if degrees_of_freedom is None
        else count(degrees_of_freedom, "degrees_of_freedom")
    )
    df = np.asarray(np.broadcast_to(df_values, observed.shape[:-1]), dtype=np.float64)
    if np.any(df <= 0):
        raise ValueError("degrees_of_freedom must be positive")
    contributions = np.zeros(observed.shape)
    positive = expected > 0
    # Divide before squaring to avoid overflowing large observed counts.
    standardized = (observed[positive] - expected[positive]) / np.sqrt(expected[positive])
    with np.errstate(over="ignore"):
        contributions[positive] = standardized**2
    if not legacy_zero_expected:
        contributions[(~positive) & (observed > 0)] = np.inf
    statistic = contributions.sum(axis=-1)
    return GoodnessOfFit(
        statistic=np.asarray(statistic),
        pvalue=np.asarray(gammaincc(df / 2, statistic / 2)),
        degrees_of_freedom=df,
        expected=expected,
        contributions=contributions,
    )
