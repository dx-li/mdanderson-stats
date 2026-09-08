"""Reachable-stage KSBIN2 design assistance and reference-completion power loss."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .ksbin2 import KSBinomialOrdering
from .ksbin2_multistage import KStageTwoSampleBinomial
from .ksbin2_probability import KSBinomialRejectionRegion
from .kstage_binomial import KStageBinomial


@dataclass(frozen=True)
class KSTwoSampleBoundaryTable:
    stage: int
    ordering: KSBinomialOrdering
    null_grid: FloatArray
    stage_null_rejection: FloatArray
    cumulative_null_rejection: FloatArray
    significance: FloatArray
    stage_power: FloatArray
    cumulative_power: FloatArray
    previous_null_rejection: FloatArray
    previous_power: FloatArray
    conditional_reference_power: FloatArray | None
    power_loss: FloatArray | None

    @property
    def midp_significance(self) -> FloatArray:
        """Original BRKARR display convention, including its first-group behavior.

        The first value halves the entire cumulative grid maximum, including
        earlier rejections. Later values average adjacent cumulative maxima.
        These displayed levels do not equal ordinary rejection probabilities.
        """
        return (self.significance + np.r_[0.0, self.significance[:-1]]) / 2

    @property
    def null_midp(self) -> FloatArray:
        """Pointwise adjustment of the current terminal group only.

        All earlier rejections retain full probability. This differs from the
        original displayed mid-p convention when prior rejection is nonzero.
        """
        preceding = np.concatenate(
            [self.previous_null_rejection[:, None], self.cumulative_null_rejection[:, :-1]], axis=1
        )
        return (self.cumulative_null_rejection + preceding) / 2

    @property
    def pointwise_midp_significance(self) -> FloatArray:
        """Grid maximum of current-group-only pointwise mid-p values."""
        return self.null_midp.max(axis=0)


def ksbin2_boundary_table(
    design: KStageTwoSampleBinomial,
    stage: int,
    probability1: ArrayLike,
    probability2: ArrayLike,
    *,
    null_grid: ArrayLike | None = None,
    reference: KSBinomialRejectionRegion | None = None,
) -> KSTwoSampleBoundaryTable:
    """Evaluate all whole tied rejection and futility groups at a reachable stage.

    Cumulative rejection includes earlier-stage rejections and the proposed
    current region, excluding future stages. Optional power loss sums paths that
    enter the candidate futility region and would reject if allowed to finish the
    reference fixed-size trial. Reference probabilities do not affect its fixed
    event region; completion is evaluated at the probabilities supplied here.
    """
    if not isinstance(design, KStageTwoSampleBinomial):
        raise ValueError("design must be a KStageTwoSampleBinomial")
    p1, p2 = np.broadcast_arrays(
        finite(probability1, "probability1"), finite(probability2, "probability2")
    )
    mass = design.stage_distribution(stage, p1, p2)
    ordering = design.orderings[stage - 1]
    grid = (
        np.arange(51, dtype=np.float64) * (0.01 if design.alternative == "two-sided" else 0.02)
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
    previous_null = np.zeros(grid.shape)
    previous_power = np.zeros(p1.shape)
    for i in range(stage - 1):
        group = design.reject_group[i]
        if group < 0:
            continue
        earlier = design.orderings[i]
        points = earlier.events[: earlier.group_end[group] + 1]
        a, b = points.T
        previous_null += design.stage_distribution(i + 1, grid, grid)[..., a, b].sum(axis=-1)
        previous_power += design.stage_distribution(i + 1, p1, p2)[..., a, b].sum(axis=-1)
    a, b = ordering.events.T
    null_mass = design.stage_distribution(stage, grid, grid)[..., a, b]
    alt_mass = mass[..., a, b]
    null = np.cumsum(null_mass, axis=-1)[..., ordering.group_end]
    power = np.cumsum(alt_mass, axis=-1)[..., ordering.group_end]
    cumulative_null = np.clip(null + previous_null[..., None], 0, 1)
    cumulative_power = np.clip(power + previous_power[..., None], 0, 1)
    conditional = None
    loss = None
    if reference is not None:
        if not isinstance(reference, KSBinomialRejectionRegion):
            raise ValueError("reference must be a KSBinomialRejectionRegion")
        final = design.cumulative_trials[-1]
        if reference.table.ordering.trials != final:
            raise ValueError("reference must use the final planned sample sizes")
        region = np.zeros((final[0] + 1, final[1] + 1))
        region[reference.events[:, 0], reference.events[:, 1]] = 1
        current = design.cumulative_trials[stage - 1]

        def transition(n: int, total: int, p: FloatArray) -> FloatArray:
            remaining = total - n
            pmf = (
                KStageBinomial([remaining], [], []).stage_distribution(1, p)
                if remaining
                else np.ones((*p.shape, 1))
            )
            needed = np.arange(total + 1)[None, :] - np.arange(n + 1)[:, None]
            return np.where(
                (needed >= 0) & (needed <= remaining), pmf[..., np.clip(needed, 0, remaining)], 0
            )

        first = transition(current[0], final[0], p1)
        second = transition(current[1], final[1], p2)
        completion = np.clip(first @ region @ np.swapaxes(second, -1, -2), 0, 1)
        conditional = completion[..., a, b]
        contribution = conditional * alt_mass
        starts = np.r_[0, ordering.group_end[:-1] + 1]
        loss = np.cumsum(contribution[..., ::-1], axis=-1)[..., ::-1][..., starts]
    return KSTwoSampleBoundaryTable(
        stage,
        ordering,
        grid,
        null,
        cumulative_null,
        np.max(cumulative_null, axis=0),
        power,
        cumulative_power,
        previous_null,
        previous_power,
        conditional,
        loss,
    )
