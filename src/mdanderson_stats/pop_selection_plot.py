"""Optional plot for the executable PoP isotonic MTD selector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._validation import scalar
from .pop_design import PoPSelection

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_pop_selection(
    selection: PoPSelection,
    *,
    target: float,
    ax: Axes | None = None,
    dose_labels: tuple[str, ...] | None = None,
) -> Axes:
    """Plot eligible isotonic estimates against the target at original dose indices.

    Untreated and safety-excluded doses are not plotted. A selected dose is
    highlighted in red. The native help mentions credible intervals, but its
    executable selector and plot do not calculate them; this plot adds none.
    """

    if not isinstance(selection, PoPSelection):
        raise TypeError("selection must be a PoPSelection")
    target_value = scalar(target, "target")
    if not 0.05 <= target_value <= 0.6:
        raise ValueError("target must be in [0.05, 0.6]")
    raw_estimate, raw_eligible = selection.isotonic_estimate, selection.eligible
    if (
        not isinstance(raw_estimate, np.ndarray)
        or not isinstance(raw_eligible, np.ndarray)
        or raw_estimate.ndim != 1
        or not 2 <= raw_estimate.size <= 100
        or raw_eligible.shape != raw_estimate.shape
        or np.iscomplexobj(raw_estimate)
        or raw_eligible.dtype.kind != "b"
    ):
        raise ValueError("selection must contain matching vectors for 2..100 doses")
    estimate = np.asarray(raw_estimate, dtype=float)
    eligible = np.asarray(raw_eligible, dtype=bool)
    if np.any(~np.isfinite(estimate[eligible])):
        raise ValueError("eligible dose estimates must be finite")
    if selection.dose is not None and not 1 <= selection.dose <= estimate.size:
        raise ValueError("selected dose is outside the estimate vector")
    if selection.dose is not None and not eligible[selection.dose - 1]:
        raise ValueError("selected dose must be marked eligible")
    if dose_labels is not None and (
        not isinstance(dose_labels, tuple)
        or len(dose_labels) != estimate.size
        or any(not isinstance(label, str) or not label for label in dose_labels)
    ):
        raise ValueError("dose_labels must be a tuple of one nonempty label per original dose")

    from matplotlib import pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    selected = eligible.copy()
    if selection.dose is not None:
        selected[selection.dose - 1] = False
        ax.scatter(
            np.arange(1, estimate.size + 1)[selection.dose - 1],
            estimate[selection.dose - 1],
            color="red",
            marker="o",
            label="Selected MTD",
            zorder=3,
        )
    if np.any(selected):
        ax.scatter(
            np.arange(1, estimate.size + 1)[selected],
            estimate[selected],
            color="black",
            marker="o",
            label="Eligible isotonic estimate",
        )
    ax.axhline(target_value, color="red", linestyle=":", label="Target")
    ax.set(xlabel="Dose level", ylabel="DLT rate", ylim=(0, 1))
    ax.set_xlim(0.5, estimate.size + 0.5)
    ax.set_xticks(np.arange(1, estimate.size + 1))
    if dose_labels is not None:
        ax.set_xticklabels(dose_labels)
    ax.legend()
    return ax
