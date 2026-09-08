"""Multiple-range comparisons from RANGE2 and KWRANGE.

The optional legacy grouping procedure is a Python adaptation of the MD Anderson
programs. See THIRD_PARTY_NOTICES.md and notices/mdanderson-range-LEGALITIES.txt.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class RangeComparisons:
    """Matrices use original input order; group indices are zero-based.

    ``order`` sorts the supplied means/rank means. ``similarity_groups`` contains
    maximal contiguous groups in that order. In the default mode, every pair in
    a group is nonsignificant. With legacy=True it reproduces the original range
    envelope groups, which can override significant interior pair comparisons.
    """

    order: NDArray[np.intp]
    statistic: FloatArray
    reject: NDArray[np.bool_]
    similarity_groups: tuple[tuple[int, ...], ...]
    critical_value: float


def _groups(values: ArrayLike, sizes: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x, n = finite(values, "values"), count(sizes, "sizes")
    if x.ndim != 1 or n.ndim != 1 or x.shape != n.shape or x.size < 2:
        raise ValueError("values and sizes must be equal-length vectors of at least two groups")
    if np.any(n == 0):
        raise ValueError("Every group size must be positive")
    return x, n


def _compare(
    values: FloatArray, sizes: FloatArray, variance: float, critical_value: float, legacy: bool
) -> RangeComparisons:
    critical_value = scalar(critical_value, "critical_value")
    if critical_value <= 0:
        raise ValueError("critical_value must be positive")
    k = values.size
    order = np.argsort(values, kind="stable")
    if legacy:
        # Preserve the archive's tie ordering as well as its unsorted sizes.
        order = np.arange(k)
        for i in range(k - 1):
            for j in range(i + 1, k):
                if values[order[i]] > values[order[j]]:
                    order[i], order[j] = order[j], order[i]
    x = values[order]
    n = sizes if legacy else sizes[order]
    se = np.sqrt(variance) * np.sqrt(0.5 / n[:, None] + 0.5 / n[None, :])
    differences = np.abs(x[:, None] - x[None, :])
    q = np.zeros((k, k))
    np.divide(differences, se, out=q, where=se > 0)
    q[(se == 0) & (differences > 0)] = np.inf
    reject = q > critical_value
    intervals: list[tuple[int, int]] = []
    if legacy:
        visited = np.eye(k, dtype=bool)
        for i in range(k - 1):
            for j in range(k - 1, i, -1):
                if visited[i, j]:
                    break
                visited[i, j] = True
                if not reject[i, j]:
                    visited[i : j + 1, i : j + 1] = True
                    reject[i : j + 1, i : j + 1] = False
                    intervals.append((i, j))
                    break
    else:
        # Each column records the last incompatible predecessor. A sliding
        # interval then identifies maximal compatible ranges in O(k^2) total.
        last_conflict = np.full(k, -1)
        row, column = np.nonzero(np.triu(reject, k=1))
        np.maximum.at(last_conflict, column, row)
        left = 0
        for right in range(k):
            next_left = max(left, int(last_conflict[right]) + 1)
            if next_left > left:
                if right - left > 1:
                    intervals.append((left, right - 1))
                left = next_left
        if k - left > 1:
            intervals.append((left, k - 1))
    # The legacy procedure writes only its upper triangle; expose a symmetric matrix.
    upper_reject = np.triu(reject, k=1)
    reject = upper_reject | upper_reject.T
    statistic = np.empty_like(q)
    decisions = np.empty_like(reject)
    statistic[np.ix_(order, order)] = q
    decisions[np.ix_(order, order)] = reject
    groups = tuple(tuple(int(v) for v in order[i : j + 1]) for i, j in intervals)
    return RangeComparisons(order, statistic, decisions, groups, critical_value)


def range2(
    means: ArrayLike,
    sizes: ArrayLike,
    error_mean_square: float,
    critical_value: float,
    *,
    legacy: bool = False,
) -> RangeComparisons:
    """Compare group means using a supplied Tukey studentized-range cutoff.

    q_ij = abs(mean_i-mean_j) / sqrt(MSE*(1/n_i+1/n_j)/2).
    Obtain the cutoff for the desired family size, error degrees of freedom and
    significance level before calling, as in the original program. Degrees of
    freedom and alpha are not used separately once that cutoff is supplied.

    The default keeps sizes with their means and tests every pair. ``legacy``
    reproduces RANGE2's failure to sort sizes and its envelope grouping rule.
    No attempt is made to reproduce uninitialized diagonal output characters.
    """
    x, n = _groups(means, sizes)
    variance = scalar(error_mean_square, "error_mean_square")
    if variance < 0:
        raise ValueError("error_mean_square must be nonnegative")
    return _compare(x, n, variance, critical_value, legacy)


def kwrange(
    ranks: ArrayLike,
    sizes: ArrayLike,
    critical_value: float,
    *,
    rank_sums: bool = False,
    legacy: bool = False,
) -> RangeComparisons:
    """KWRANGE's rank-based comparisons with a caller-supplied critical value.

    Input group mean ranks, or rank sums with ``rank_sums=True``. This function
    retains the archive's denominator sqrt(N*(N+1)*(1/n_i+1/n_j)/12), where N is
    total sample size. This is sqrt(2) larger than a conventional Nemenyi
    studentized-range denominator: do not interchange those critical values.
    There is no tie-variance correction, matching the original.

    The default keeps sizes with their groups and tests every pair. ``legacy``
    also reproduces the archive's size-sorting defect and envelope grouping.
    """
    x, n = _groups(ranks, sizes)
    if rank_sums:
        x = x / n
    total = float(n.sum())
    variance = total * (total + 1) / 6
    if not np.isfinite(variance):
        raise ValueError("Total sample size is too large")
    return _compare(x, n, variance, critical_value, legacy)
