"""KSBIN2's two-sample binomial evidence statistics and outcome ordering."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count


def ksbin2_statistic(
    events1: ArrayLike,
    trials1: ArrayLike,
    events2: ArrayLike,
    trials2: ArrayLike,
    *,
    criteria: tuple[int, ...] = (1, 2),
    alternative: str = "greater",
) -> FloatArray:
    """Return the source's weighted score; smaller favors the alternative.

    Criteria: 1 difference, 2 log likelihood ratio, 3 Pearson chi-square,
    4 absolute unpooled Z (999 for unequal proportions with zero variance).
    'greater' means p1 > p2, 'less' p1 < p2; 'two-sided' uses either direction.
    Unique criteria combine with successive weights 0.001**position, not a
    lexicographic sort. Counts broadcast; each group has 1–100 observations.
    """
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError("alternative must be less, greater or two-sided")
    kinds = count(criteria, "criteria")
    if (
        kinds.ndim != 1
        or not 1 <= len(kinds) <= 4
        or np.any((kinds < 1) | (kinds > 4))
        or len(np.unique(kinds)) != len(kinds)
    ):
        raise ValueError("criteria requires 1–4 distinct integers from 1 to 4")
    k1, n1, k2, n2 = np.broadcast_arrays(
        count(events1, "events1"),
        count(trials1, "trials1"),
        count(events2, "events2"),
        count(trials2, "trials2"),
    )
    if np.any((n1 < 1) | (n1 > 100) | (n2 < 1) | (n2 > 100) | (k1 > n1) | (k2 > n2)):
        raise ValueError("group sizes must be 1–100 and events cannot exceed trials")
    p1, p2 = k1 / n1, k2 / n2
    difference = np.abs(p1 - p2)
    equal = k1 * n2 == k2 * n1
    score = np.zeros(k1.shape)

    def maximum_loglik(k: FloatArray, n: FloatArray) -> FloatArray:
        p = k / n
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(k == 0, 0, k * np.log(p)) + np.where(k == n, 0, (n - k) * np.log1p(-p))

    for position, kind in enumerate(kinds):
        if kind == 1:
            value = difference
        elif kind == 2:
            value = (
                maximum_loglik(k1, n1) + maximum_loglik(k2, n2) - maximum_loglik(k1 + k2, n1 + n2)
            )
            # The exact likelihood ratio is one for equal observed proportions.
            value = np.where(equal, 0, np.maximum(value, 0))
        elif kind == 3:
            numerator = (k1 * n2 - k2 * n1) ** 2 * (n1 + n2)
            denominator = n1 * n2 * (k1 + k2) * (n1 + n2 - k1 - k2)
            value = np.zeros(k1.shape)
            np.divide(numerator, denominator, out=value, where=denominator > 0)
        else:
            sd = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            value = np.where(equal, 0.0, 999.0)
            np.divide(difference, sd, out=value, where=sd > 0)
        score -= (0.001**position) * value
    if alternative != "two-sided":
        reverse = p1 < p2 if alternative == "greater" else p1 > p2
        score = np.where(reverse, -score, score)
    return score


@dataclass(frozen=True)
class KSBinomialOrdering:
    trials: tuple[int, int]
    criteria: tuple[int, ...]
    alternative: str
    events: np.ndarray
    score: FloatArray
    group_end: np.ndarray


def ksbin2_ordering(
    trials1: int,
    trials2: int,
    *,
    criteria: tuple[int, ...] = (1, 2),
    alternative: str = "greater",
) -> KSBinomialOrdering:
    """Order all count pairs and identify inclusive ends of tied score groups.

    events has two columns. Scores ascend (strongest evidence first); ties use
    the source's adjacent-score relative tolerance 1e-12. group_end contains
    zero-based inclusive row indices. Within ties, stable row-major input order
    is retained; the original quicksort need not retain that order.
    """
    sizes = count([trials1, trials2], "trials")
    if sizes.shape != (2,) or np.any((sizes < 1) | (sizes > 100)):
        raise ValueError("trials1 and trials2 must be scalar integers from 1 to 100")
    n1, n2 = map(int, sizes)
    k1 = np.repeat(np.arange(n1 + 1), n2 + 1)
    k2 = np.tile(np.arange(n2 + 1), n1 + 1)
    score = ksbin2_statistic(k1, n1, k2, n2, criteria=criteria, alternative=alternative)
    order = np.argsort(score, kind="stable")
    score = score[order]
    relative = np.abs(np.diff(score)) / np.maximum(np.abs(score[:-1]) + np.abs(score[1:]), 1e-100)
    ends = np.r_[np.flatnonzero(relative >= 1e-12), len(score) - 1]
    return KSBinomialOrdering(
        (n1, n2),
        tuple(map(int, criteria)),
        alternative,
        np.column_stack([k1[order], k2[order]]),
        score,
        ends,
    )
