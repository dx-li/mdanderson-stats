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

    @property
    def midp_significance(self) -> FloatArray:
        """Source BRKARR mid-p convention: average adjacent grid maxima.

        This is not the maximum of pointwise mid-p values, nor the actual
        rejection probability of the inclusive region.
        """
        return (self.significance + np.r_[0.0, self.significance[:-1]]) / 2

    @property
    def null_midp(self) -> FloatArray:
        """Pointwise strict-tail probability plus half the terminal tied mass."""
        previous = np.concatenate(
            [np.zeros((len(self.null_grid), 1)), self.null_rejection[:, :-1]], axis=1
        )
        return (self.null_rejection + previous) / 2

    @property
    def pointwise_midp_significance(self) -> FloatArray:
        """Maximum over the grid after the pointwise mid-p adjustment."""
        return self.null_midp.max(axis=0)

    def select(
        self, alpha: float = 0.05, *, method: str = "ordinary"
    ) -> "KSBinomialRejectionRegion":
        """Largest complete tied region within the chosen reported level.

        method='midp' uses KSBIN2's source convention. The returned ordinary
        significance exposes the inclusive region's actual grid-maximum size.
        alpha is scalar; alternative-power cases retain their broadcast shape.
        """
        level = finite(alpha, "alpha")
        if level.ndim != 0 or not 0 <= level <= 1:
            raise ValueError("alpha must be a scalar in [0,1]")
        if method not in ("ordinary", "midp"):
            raise ValueError("method must be ordinary or midp")
        values = self.significance if method == "ordinary" else self.midp_significance
        group = int(np.searchsorted(values, level, side="right")) - 1
        return self.select_group(group, method=method)

    def select_group(self, group: int, *, method: str = "ordinary") -> "KSBinomialRejectionRegion":
        """Choose a zero-based tied group explicitly; -1 selects the empty region."""
        if (
            isinstance(group, bool)
            or not isinstance(group, (int, np.integer))
            or not -1 <= group < len(self.significance)
        ):
            raise ValueError("group must be -1 or a valid zero-based tied group index")
        if method not in ("ordinary", "midp"):
            raise ValueError("method must be ordinary or midp")
        end = -1 if group == -1 else int(self.ordering.group_end[group])
        empty = group == -1
        return KSBinomialRejectionRegion(
            self,
            int(group),
            end,
            method,
            0.0
            if empty
            else float(
                self.significance[group] if method == "ordinary" else self.midp_significance[group]
            ),
            0.0 if empty else float(self.significance[group]),
            np.zeros(self.power.shape[:-1]) if empty else self.power[..., group].copy(),
            self.ordering.events[: end + 1].copy(),
        )


@dataclass(frozen=True)
class KSBinomialRejectionRegion:
    """An inclusive region; mid-p reporting never randomizes its terminal group."""

    table: KSBinomialProbabilityTable
    group: int
    last_row: int
    method: str
    reported_significance: float
    significance: float
    power: FloatArray
    events: np.ndarray


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
