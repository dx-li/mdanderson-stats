"""Structured summaries and text reports for MUHAZ bandwidth-selected fits."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .muhaz import Boundary, Kernel
from .muhaz_global import MuhazGlobal
from .muhaz_knn import MuhazKNN
from .muhaz_local import MuhazLocal
from .muhaz_neighbors import NeighborMethod


@dataclass(frozen=True)
class MuhazSummary:
    n_observations: int
    n_censored: int
    method: Literal["global", "local", "knn"]
    boundary: Boundary
    kernel: Kernel
    bounds: tuple[float, float]
    n_min_grid: int | None
    n_est_grid: int
    pilot_bandwidth: float | None
    smoothing_bandwidth: float | None
    constant_bandwidth: float | None
    neighbors: int | None
    neighbor_method: NeighborMethod | None
    score: float | None
    converged_cells: int
    diagnostic_cells: int
    legacy: bool

    def report(self, *, digits: int = 6) -> str:
        """Format actual settings, estimates and convergence without rounding to zero."""
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 17
        ):
            raise ValueError("digits must be an integer from 1 to 17")

        def number(value: float | None) -> str:
            return "not used" if value is None else format(value, f".{digits}g")

        rows = [
            ("Number of observations", str(self.n_observations)),
            ("Censored observations", str(self.n_censored)),
            (
                "Method",
                {"global": "Global", "local": "Local", "knn": "Nearest Neighbor"}[self.method],
            ),
            (
                "Boundary correction",
                {"none": "None", "left": "Left Only", "both": "Left and Right"}[self.boundary],
            ),
            ("Kernel", self.kernel.capitalize()),
            ("Minimum time", number(self.bounds[0])),
            ("Maximum time", number(self.bounds[1])),
            (
                "Minimization grid points",
                "not retained" if self.n_min_grid is None else str(self.n_min_grid),
            ),
            ("Estimation grid points", str(self.n_est_grid)),
            ("Pilot bandwidth", number(self.pilot_bandwidth)),
            ("Smoothing bandwidth", number(self.smoothing_bandwidth)),
            ("Legacy conventions", "yes" if self.legacy else "no"),
        ]
        if self.constant_bandwidth is not None:
            rows.append(("Constant bandwidth", number(self.constant_bandwidth)))
        if self.neighbors is not None:
            rows.extend(
                [
                    ("Selected neighbors", str(self.neighbors)),
                    ("Neighbor method", str(self.neighbor_method)),
                ]
            )
        if self.score is None:
            rows.append(("MSE selection", "bypassed (single candidate)"))
        else:
            label = "Sum of pointwise minimum MSE" if self.method == "local" else "Summed grid MSE"
            rows.extend(
                [
                    (label, number(self.score)),
                    (
                        "Converged candidate/grid cells",
                        f"{self.converged_cells}/{self.diagnostic_cells}",
                    ),
                ]
            )
        return "\n".join(f"{key}: {value}" for key, value in rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 6) -> None:
        Path(path).write_text(self.report(digits=digits), encoding="utf-8")


def summarize_muhaz(fit: MuhazGlobal | MuhazLocal | MuhazKNN) -> MuhazSummary:
    """Summarize original MUHAZ output fields plus explicit compatibility/convergence.

    Scores describe candidate minimization grids, not the smoothed final curve.
    A bypass has no computed score. Counts reflect the selected input sample.
    """
    if not isinstance(fit, (MuhazGlobal, MuhazLocal, MuhazKNN)):
        raise TypeError("fit must be a global, local or nearest-neighbor MUHAZ fit")
    curve = fit.curve if isinstance(fit, MuhazGlobal) else fit
    diagnostic = fit.diagnostics
    pilot = fit.pilot_bandwidth
    n_min = fit.n_min_grid
    if diagnostic is not None:
        pilot = diagnostic.pilot_bandwidth
        n_min = diagnostic.time.size
    constant = None
    smooth = None
    neighbors = None
    neighbor_method = None
    method: Literal["global", "local", "knn"]
    if isinstance(fit, MuhazGlobal):
        method = "global"
        constant = fit.bandwidth
    elif isinstance(fit, MuhazLocal):
        method = "local"
        smooth = fit.smoothing_bandwidth
        if fit.local_bandwidth is None:
            constant = float(fit.bandwidth[0])
    else:
        method = "knn"
        smooth = fit.smoothing_bandwidth
        neighbors = fit.neighbors
        neighbor_method = fit.neighbor_bandwidths.method
        n_min = fit.neighbor_bandwidths.time.size
    return MuhazSummary(
        fit.n_observations,
        fit.n_observations - fit.n_events,
        method,
        curve.boundary,
        curve.kernel,
        curve.bounds,
        n_min,
        fit.time.size,
        pilot,
        smooth,
        constant,
        neighbors,
        neighbor_method,
        fit.score,
        0 if diagnostic is None else int(diagnostic.converged.sum()),
        0 if diagnostic is None else diagnostic.converged.size,
        curve.legacy,
    )
