"""Optional side-by-side original and adjusted PCoA plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .apcoa import AdjustedPCoA

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_adjusted_pcoa(
    result: AdjustedPCoA,
    groups: ArrayLike | None = None,
) -> tuple[Axes, Axes]:
    """Plot the first two positive axes, colored by supplied text group labels.

    Requires the plot extra; returns axes without showing/saving the figure.
    Axis percentages use positive inertia, not the possibly negative signed trace.
    Native ellipses and medoid connectors are not reproduced.
    """
    if not isinstance(result, AdjustedPCoA):
        raise TypeError("result must be AdjustedPCoA")
    if min(result.original.coordinates.shape[1], result.adjusted.coordinates.shape[1]) < 2:
        raise ValueError("both ordinations require at least two positive coordinate axes")
    n = result.original.coordinates.shape[0]
    labels = np.full(n, "All samples") if groups is None else np.asarray(groups)
    if labels.shape != (n,) or not all(isinstance(s, str) and s for s in labels):
        raise ValueError("groups must be one nonempty text label per sample")
    from matplotlib import pyplot as plt

    _, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, fit, title in zip(
        axes, [result.original, result.adjusted], ["Original PCoA", "Adjusted PCoA"]
    ):
        for label in np.unique(labels):
            selected = labels == label
            ax.scatter(fit.coordinates[selected, 0], fit.coordinates[selected, 1], label=label)
        ax.set_title(title)
        ax.set_xlabel(f"Axis 1 ({100 * fit.positive_fraction[0]:.1f}% positive inertia)")
        ax.set_ylabel(f"Axis 2 ({100 * fit.positive_fraction[1]:.1f}% positive inertia)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[1].legend()
    return axes[0], axes[1]
