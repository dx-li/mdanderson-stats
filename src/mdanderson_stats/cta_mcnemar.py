"""CTA pairwise McNemar, pooled direction and heterogeneity decomposition."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc

from ._validation import FloatArray, finite


@dataclass(frozen=True)
class McNemarAnalysis:
    pairs: NDArray[np.int64]
    above: FloatArray
    below: FloatArray
    pair_statistic: FloatArray
    pair_pvalue: FloatArray
    summed_statistic: FloatArray
    summed_pvalue: FloatArray
    summed_df: NDArray[np.int64]
    pooled_statistic: FloatArray
    pooled_pvalue: FloatArray
    pooled_df: NDArray[np.int64]
    heterogeneity_statistic: FloatArray
    heterogeneity_pvalue: FloatArray
    heterogeneity_df: NDArray[np.int64]


def mcnemar_analysis(observed: ArrayLike) -> McNemarAnalysis:
    """Analyze square paired-category tables along the last two axes.

    For each i<j, compare O[i,j] against O[j,i] without continuity correction.
    Zero-discordance pairs contribute zero and are omitted from degrees of
    freedom. Undefined zero-df p-values are NaN, including heterogeneity in 2x2.
    The pooled direction uses category ordering; this is not Stuart-Maxwell's
    marginal-homogeneity test. Fractional nonnegative frequencies are allowed.
    """
    ob = finite(observed, "observed")
    if ob.ndim < 2 or ob.shape[-1] != ob.shape[-2] or ob.shape[-1] < 2 or np.any(ob < 0):
        raise ValueError("observed must contain nonnegative square tables of size at least two")
    i, j = np.triu_indices(ob.shape[-1], 1)
    pairs = np.column_stack((i, j)).astype(np.int64)
    above, below = ob[..., i, j].copy(), ob[..., j, i].copy()
    with np.errstate(over="ignore"):
        weights = above + below
        total = weights.sum(axis=-1)
    if not np.all(np.isfinite(total)):
        raise ValueError("total discordant frequency must be finite")
    active = weights > 0
    difference = above - below
    ratio = np.divide(difference, weights, out=np.zeros_like(weights), where=active)
    # Both terms are bounded by the discordant counts, avoiding raw squaring.
    pair_stat = difference * ratio
    summed = np.asarray(pair_stat.sum(axis=-1))
    directional_sum = difference.sum(axis=-1)
    pooled_ratio = np.divide(directional_sum, total, out=np.zeros_like(total), where=total > 0)
    pooled = np.asarray(directional_sum * pooled_ratio)
    # Stable equivalent of summed-pooled, without cancellation near homogeneity.
    hetero = np.asarray(np.sum(weights * (ratio - pooled_ratio[..., None]) ** 2, axis=-1))
    summed_df = np.asarray(active.sum(axis=-1), dtype=np.int64)
    pooled_df = np.asarray(total > 0, dtype=np.int64)
    hetero_df = np.asarray(np.maximum(summed_df - 1, 0), dtype=np.int64)
    pair_p = np.where(active, gammaincc(0.5, pair_stat / 2), np.nan)
    summed_p = np.where(summed_df > 0, gammaincc(summed_df / 2, summed / 2), np.nan)
    pooled_p = np.where(pooled_df > 0, gammaincc(0.5, pooled / 2), np.nan)
    hetero_p = np.where(hetero_df > 0, gammaincc(hetero_df / 2, hetero / 2), np.nan)
    for array in (
        pairs,
        above,
        below,
        pair_stat,
        pair_p,
        summed,
        summed_p,
        summed_df,
        pooled,
        pooled_p,
        pooled_df,
        hetero,
        hetero_p,
        hetero_df,
    ):
        array.flags.writeable = False
    return McNemarAnalysis(
        pairs,
        above,
        below,
        pair_stat,
        pair_p,
        summed,
        summed_p,
        summed_df,
        pooled,
        pooled_p,
        pooled_df,
        hetero,
        hetero_p,
        hetero_df,
    )
