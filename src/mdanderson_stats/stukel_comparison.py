"""The archived STUKEL demo's six-family comparison on supplied or example data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .stukel_fit import StukelFit, StukelFitError, fit_stukel
from .stukel_output import format_stukel

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_LABELS = ("Standard", "Alpha1 only", "Alpha2 only", "Equal shapes", "Opposite shapes", "Both free")


@dataclass(frozen=True)
class StukelComparison:
    """Six fitted families in order 0..5, with paired single-dose observations."""

    x: FloatArray
    successes: FloatArray
    trials: FloatArray
    fits: tuple[StukelFit, ...]

    def report(self, *, digits: int = 4) -> str:
        """Return all six regression tables in family order, without file writes."""
        return "\n".join(format_stukel(fit, digits=digits) for fit in self.fits)

    def plot(self) -> tuple[Axes, Axes]:
        """Draw dose-response and baseline-log-odds comparison panels.

        Requires the plot extra. As in demos.S, all six transformed log odds in
        the right panel are plotted against family 0's fitted log odds, not each
        family's own linear predictor. Returns axes without showing or saving.
        """
        from matplotlib import pyplot as plt

        _, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        dose, link = axes
        order = np.argsort(self.x, kind="stable")
        baseline = self.fits[0].objective.log_odds
        link_order = np.argsort(baseline, kind="stable")
        dose.plot(
            self.x[order],
            (self.successes / self.trials)[order],
            "o",
            markerfacecolor="none",
            color="black",
            label="Observed",
        )
        for index, (fit, label) in enumerate(zip(self.fits, _LABELS, strict=True)):
            # The same color and style identify a family in both panels.
            style = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))[index]
            dose.plot(
                self.x[order],
                fit.objective.probabilities[order],
                label=label,
                color=f"C{index}",
                linestyle=style,
            )
            link.plot(
                baseline[link_order],
                fit.objective.log_odds[link_order],
                label=label,
                color=f"C{index}",
                linestyle=style,
            )
        dose.set(xlabel="Dose", ylabel="Probability", title="Estimated dose-response curves")
        link.set(
            xlabel="Standard-model fitted log odds",
            ylabel="Fitted log odds h",
            title="Six-family link comparison",
        )
        dose.legend(loc="best", fontsize="small")
        link.legend(loc="best", fontsize="small")
        return dose, link


def compare_stukel(
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    scale: str = "pearson",
    tolerance: float = 1e-12,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> StukelComparison:
    """Fit all six families with an intercept and original zero shape starts.

    x is one covariate, a vector or single-column matrix. Each model uses the
    same supplied observations. A failed fit raises with its family number;
    comparisons never quietly omit a family or automatically choose a winner.
    """
    dose = finite(x, "x")
    if dose.ndim == 2 and dose.shape[1] == 1:
        dose = dose[:, 0]
    if dose.ndim != 1 or dose.size == 0:
        raise ValueError("Comparison requires one nonempty dose covariate")
    y, n = count(successes, "successes"), count(trials, "trials")
    if y.shape != dose.shape or n.shape != y.shape or np.any(n == 0) or np.any(y > n):
        raise ValueError("Counts must match doses with 0 <= successes <= positive trials")
    fitted = []
    for family in range(6):
        try:
            fitted.append(
                fit_stukel(
                    dose,
                    y,
                    n,
                    family=family,
                    scale=scale,
                    tolerance=tolerance,
                    gradient_tolerance=gradient_tolerance,
                    max_iterations=max_iterations,
                )
            )
        except ValueError as error:
            raise StukelFitError(f"Comparison family {family}: {error}") from error
    snapshots = [array.copy() for array in (dose, y, n)]
    for array in snapshots:
        array.flags.writeable = False
    return StukelComparison(snapshots[0], snapshots[1], snapshots[2], tuple(fitted))


def stukel_demo(dataset: str = "beetles") -> StukelComparison:
    """Run the original six-family demo on 'beetles' or 'warsaw'.

    The archived numeric example data are packaged locally. No network, terminal
    prompt, plot window, or file write occurs; call result.report()/plot().
    """
    if dataset not in ("beetles", "warsaw"):
        raise ValueError("dataset must be 'beetles' or 'warsaw'")
    record = json.loads(files("mdanderson_stats").joinpath("stukel_examples.json").read_text())
    data = record[dataset]
    return compare_stukel(data["x"], data["successes"], data["trials"])
