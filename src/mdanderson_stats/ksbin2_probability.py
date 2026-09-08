"""Single-stage KSBIN2 power and null-grid significance by tied rejection group."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .ksbin2 import KSBinomialOrdering, ksbin2_ordering
from .kstage_binomial import KStageBinomial


@dataclass(frozen=True)
class KSBinomialProbabilityTable:
    """Each last-axis entry includes one complete tied rejection group.

    significance is a maximum over null_grid only, not a continuum supremum.
    null_rejection has shape (grid points, groups); power follows the broadcast
    alternative probability shape with a final group axis. All regions are
    ordinary inclusive nonrandomized regions; no mid-p adjustment is applied.
    """

    ordering: KSBinomialOrdering
    probability1: FloatArray
    probability2: FloatArray
    null_grid: FloatArray
    null_rejection: FloatArray
    significance: FloatArray
    maximizing_null_probability: FloatArray
    power: FloatArray


def ksbin2_probability_table(
    trials1: int,
    trials2: int,
    probability1: ArrayLike,
    probability2: ArrayLike,
    *,
    criteria: tuple[int, ...] = (1, 2),
    alternative: str = "greater",
    null_grid: ArrayLike | None = None,
) -> KSBinomialProbabilityTable:
    """Calculate power and grid-maximum significance for all inclusive regions.

    Default null grids match KSBIN2: 0:0.02:1 for one-sided tests, 0:0.01:0.5
    for two-sided tests. A custom increasing grid in [0,1] is allowed. Every
    region includes all outcome rows through ordering.group_end[g]. Values are
    computed for the entire table without the source's significance cutoff.
    """
    ordering = ksbin2_ordering(trials1, trials2, criteria=criteria, alternative=alternative)
    p1, p2 = np.broadcast_arrays(
        finite(probability1, "probability1"), finite(probability2, "probability2")
    )
    if np.any((p1 < 0) | (p1 > 1) | (p2 < 0) | (p2 > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    if (alternative == "greater" and np.any(p1 <= p2)) or (
        alternative == "less" and np.any(p1 >= p2)
    ):
        raise ValueError("probabilities must follow the specified alternative direction")
    grid = (
        np.arange(51, dtype=np.float64) * (0.01 if alternative == "two-sided" else 0.02)
        if null_grid is None
        else finite(null_grid, "null_grid")
    )
    if (
        grid.ndim != 1
        or grid.size == 0
        or np.any((grid < 0) | (grid > 1))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("null_grid must be a nonempty strictly increasing vector in [0,1]")
    n1, n2 = ordering.trials
    first, second = KStageBinomial([n1], [], []), KStageBinomial([n2], [], [])
    k1, k2 = ordering.events.T

    def rejection(a: FloatArray, b: FloatArray) -> FloatArray:
        mass = first.stage_distribution(1, a)[..., k1] * second.stage_distribution(1, b)[..., k2]
        return np.clip(np.cumsum(mass, axis=-1)[..., ordering.group_end], 0, 1)

    null = rejection(grid, grid)
    maximum = np.argmax(null, axis=0)
    return KSBinomialProbabilityTable(
        ordering, p1, p2, grid, null, np.max(null, axis=0), grid[maximum], rejection(p1, p2)
    )
