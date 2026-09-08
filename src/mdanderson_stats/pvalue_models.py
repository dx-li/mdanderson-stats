"""MULTI order-statistic diagnostics and clustered p-value simulation.

Adapted from OSFIT and CLUSTP; see notices/mdanderson-multi-LEGALITIES.txt.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, ndtr

from ._validation import FloatArray, scalar
from .multiplicity import _pvalues


@dataclass(frozen=True)
class OrderStatisticDiagnostics:
    """Diagnostic arrays in input order, with sorted-rank indices in order.

    combined_score is the uncapped OSFIT score, not a posterior probability.
    legacy_transformed_cdf preserves the source's current-rank denominator.
    """

    cumulative: FloatArray
    legacy_transformed_cdf: FloatArray
    combined_score: FloatArray
    order: NDArray[np.intp]


def order_statistic_diagnostics(pvalues: ArrayLike) -> OrderStatisticDiagnostics:
    """Evaluate MULTI's OSFIT on each family along the last axis.

    For sorted x(i), cumulative is I_x(i, n-i+1). The legacy transformed
    diagnostic is I_q(1, n-i+1), with q=x(1) for the first rank and
    q=min(x(i)/(1-x(i)), 1) thereafter. This follows the executable source,
    whose denominator differs from the accompanying written description.
    combined_score = n*min(cumulative) is deliberately not capped at one.
    """
    values = _pvalues(pvalues)
    order = np.argsort(values, axis=-1, kind="stable")
    x = np.take_along_axis(values, order, axis=-1)
    n = x.shape[-1]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    remaining = n + 1 - ranks
    cumulative = betainc(ranks, remaining, x)
    # Ratios at or above 1/2 are capped before division, including x=1.
    q = np.ones_like(x)
    np.divide(x, 1 - x, out=q, where=x < 0.5)
    q[..., 0] = x[..., 0]
    with np.errstate(divide="ignore"):
        transformed = -np.expm1(remaining * np.log1p(-q))
    inverse = np.argsort(order, axis=-1)
    return OrderStatisticDiagnostics(
        np.take_along_axis(cumulative, inverse, axis=-1),
        np.take_along_axis(transformed, inverse, axis=-1),
        np.asarray(n * np.min(cumulative, axis=-1)),
        order,
    )


def clustered_pvalues(
    null_clusters: int,
    alternative_clusters: int = 0,
    *,
    cluster_size: int = 1,
    correlation: float = 0.0,
    alternative_mean: float = 1.96,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Generate CLUSTP's one-sided normal p-values, one cluster per row.

    Null rows precede alternative rows. Correlation specifies the latent normal
    statistics' equicorrelation, not the transformed p-values' Pearson correlation.
    For k>1, require -1/(k-1) < correlation < 1 (positive definite covariance).
    For k=1, correlation is irrelevant but must be in [-1,1]. Unlike the original
    fixed workspace, cluster_size is not capped at 100. NumPy random streams
    replace the source's global single-precision generator.
    """
    for name, value, minimum in (
        ("null_clusters", null_clusters, 0),
        ("alternative_clusters", alternative_clusters, 0),
        ("cluster_size", cluster_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rho = scalar(correlation, "correlation")
    mean = scalar(alternative_mean, "alternative_mean")
    if not -1 <= rho <= 1:
        raise ValueError("correlation must be in [-1, 1]")
    k = cluster_size
    if k > 1 and not -1 / (k - 1) < rho < 1:
        raise ValueError("correlation must give a positive definite covariance matrix")
    generator = np.random.default_rng(rng)
    z = generator.standard_normal((null_clusters + alternative_clusters, k))
    if k > 1:
        # Orthogonal mean/contrast projections diagonalize equicorrelation.
        # This is O(rows*k), without allocating or factoring a k-by-k matrix.
        row_mean = z.mean(axis=-1, keepdims=True)
        z -= row_mean
        z *= np.sqrt(1 - rho)
        z += np.sqrt(1 + (k - 1) * rho) * row_mean
    z[null_clusters:] += mean
    return np.asarray(ndtr(-z))
