"""Optional prior/posterior history display for the beta-binomial learning demo."""

from typing import TYPE_CHECKING

import numpy as np

from .bayesian_monitoring import _integer
from .beta_binomial import BetaBinomialPosterior, BetaBinomialSequence

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_beta_binomial_sequence(
    sequence: BetaBinomialSequence,
    *,
    cohort: int | None = None,
    trial_index: tuple[int, ...] = (),
    credible_probability: float = 0.95,
    points: int = 1001,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot one selected history stage; cohort zero displays the original prior.

    Leading sequence axes are selected explicitly with trial_index. Earlier curves
    are gray, the immediate prior blue and the current posterior red. Credible
    regions are calculated analytically, independently of the drawing grid.
    Imports Matplotlib lazily; does not show, save, clear axes or change backend.
    """
    shape = sequence.posterior.alpha.shape
    if len(trial_index) != len(shape) - 1:
        raise ValueError("trial_index must select every leading sequence axis")
    for index, length in zip(trial_index, shape[:-1], strict=True):
        if not 0 <= _integer(index, "trial_index") < length:
            raise ValueError("trial_index is out of range")
    step = shape[-1] - 1 if cohort is None else _integer(cohort, "cohort")
    if not 0 <= step < shape[-1]:
        raise ValueError("cohort must lie between zero and the number of cohorts")
    resolution = _integer(points, "points")
    if not 101 <= resolution <= 100001:
        raise ValueError("points must lie in [101,100001]")
    aa = sequence.posterior.alpha[trial_index][: step + 1]
    bb = sequence.posterior.beta[trial_index][: step + 1]
    current = BetaBinomialPosterior(aa[-1], bb[-1])
    regions = current.credible_set(credible_probability)
    bounds = regions.intervals[np.isfinite(regions.intervals[:, 0])]
    from matplotlib import pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    singular = False
    for i, (a, b) in enumerate(zip(aa, bb, strict=True)):
        distribution = BetaBinomialPosterior(a, b)
        # Supplement a uniform grid to resolve concentrated posterior peaks.
        x = np.unique(
            np.r_[
                np.linspace(0, 1, resolution), distribution.quantile(np.linspace(0.001, 0.999, 201))
            ]
        )
        y = distribution.pdf(x)
        if np.any(~np.isfinite(y[(x > 0) & (x < 1)])):
            raise ArithmeticError("interior beta density is not representable for plotting")
        singular |= bool(np.any(np.isinf(y)))
        y[~np.isfinite(y)] = np.nan  # Infinite boundary densities cannot be drawn.
        color = "#bb2525" if i == step else "#1765ad" if i == step - 1 else "0.72"
        label = (
            ("Current posterior" if step else "Original prior")
            if i == step
            else "Immediate prior"
            if i == step - 1
            else "Earlier distributions"
            if i == 0
            else None
        )
        ax.plot(x, y, color=color, linewidth=1.8 if i >= step - 1 else 1, label=label)
    for j, (lo, hi) in enumerate(bounds):
        ax.axvspan(
            lo,
            hi,
            color="#bb2525",
            alpha=0.08,
            label=f"{100 * credible_probability:g}% highest-density set" if j == 0 else None,
        )
    interval_text = " or ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in bounds)
    if step:
        s, f = sequence.successes[trial_index][step - 1], sequence.failures[trial_index][step - 1]
        cs, cf = (
            sequence.cumulative_successes[trial_index][step - 1],
            sequence.cumulative_failures[trial_index][step - 1],
        )
        detail = f"Cohort {step}: {s:g} successes, {f:g} failures; cumulative {cs:g}, {cf:g}"
    else:
        detail = "Before observations"
    ax.set(
        xlim=(0, 1),
        xlabel="Success probability",
        ylabel="Probability density",
        title=(
            f"{detail}\nBeta({aa[-1]:g}, {bb[-1]:g}); mean = {float(current.mean):.4g}"
            f"\nCredible set: {interval_text}"
        ),
    )
    ax.set_ylim(bottom=0)
    if singular:
        ax.text(
            0.02,
            0.98,
            "Unbounded endpoint density omitted",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    ax.legend(loc="best", fontsize=8)
    return ax
