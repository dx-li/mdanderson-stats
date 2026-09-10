"""Dependent order-statistic bounds for posterior chi-square diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc

from ._validation import FloatArray, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class ChiSquareOrderBounds:
    ranks: NDArray[np.int64]
    statistics: FloatArray
    fixed_rank_bound: FloatArray
    minimum_diagnostic: float
    minimizing_rank: int
    search_adjusted_bound: float
    posterior_samples: int
    degrees_of_freedom: float
    upper_trim: float


def chi_square_order_bounds(
    statistics: ArrayLike,
    degrees_of_freedom: float,
    *,
    upper_trim: float = 0.005,
) -> ChiSquareOrderBounds:
    """Bound P(D_(r)>t) by min(1, J*chi_square_sf(t)/(J-r+1)).

    Ascending ranks are one-based. No independence between posterior draws is
    assumed; the bound requires the stated marginal reference distribution.
    The minimum over ranks is a diagnostic, not a globally calibrated p-value.
    search_adjusted_bound applies a Bonferroni correction over the retained ranks.
    upper_trim excludes ceil(J*upper_trim) upper order statistics, retaining at
    least one rank. .005 follows Yuan and Johnson's diagnostic search convention;
    BCSTTE's exact native rank/trim convention remains unverified.
    """
    values = finite(statistics, "statistics")
    df = scalar(degrees_of_freedom, "degrees_of_freedom")
    trim = scalar(upper_trim, "upper_trim")
    if values.ndim != 1 or not 1 <= values.size <= 1000000 or np.any(values < 0):
        raise ValueError("require 1..1000000 nonnegative statistics")
    if df / 2 <= 0 or not 0 <= trim < 1:
        raise ValueError("require positive degrees of freedom and upper_trim in [0,1)")
    total = values.size
    retained = max(1, total - int(np.ceil(total * trim)))
    ordered = np.sort(values)[:retained]
    ranks = np.arange(1, retained + 1, dtype=np.int64)
    # Retain a positive conservative floor instead of reporting a zero bound
    # when the reference survival function underflows.
    tails = np.maximum(gammaincc(df / 2, ordered / 2), np.finfo(float).tiny)
    bounds = np.minimum(1.0, tails * (total / (total - ranks + 1)))
    best = int(np.argmin(bounds))
    minimum = float(bounds[best])
    return ChiSquareOrderBounds(
        _owned(ranks),
        _owned(ordered),
        _owned(bounds),
        minimum,
        int(ranks[best]),
        min(1.0, retained * minimum),
        total,
        df,
        trim,
    )
