"""MULTI S betamix sequential component selection; see the MULTI notice."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import scalar
from .beta_mixture import BetaMixture
from .beta_mixture_bootstrap import (
    BetaMixtureBootstrap,
    _controls,
    _fit,
    _positive_integer,
    beta_mixture_bootstrap,
)
from .beta_mixture_fit import BetaMixtureFit, BetaMixtureFitError, beta_mixture_start
from .multiplicity import _pvalues


@dataclass(frozen=True)
class BetaMixtureSelection:
    fit: BetaMixtureFit
    candidates: tuple[BetaMixtureFit, ...]
    bootstrap_checks: tuple[BetaMixtureBootstrap, ...]
    criterion: str
    status: str
    message: str


def select_beta_mixture(
    pvalues: ArrayLike,
    *,
    criterion: str = "data",
    threshold: float = 0.05,
    algorithm: str = "ml",
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    max_components: int = 10,
    legacy_endpoints: bool = False,
    replicates: int = 100,
    rng: np.random.Generator | int | None = None,
) -> BetaMixtureSelection:
    """Select k=0..10 by new weight, relative likelihood change, or simulated CVM.

    data/lglk select the preceding model when the strict threshold is met;
    pcvm selects the current model when its simulated p-value exceeds threshold.
    Nonconvergence returns the preceding valid fit with explicit status/message,
    matching the source's warning-bearing return rather than claiming selection.
    """
    x = _pvalues(pvalues)
    if x.ndim != 1:
        raise ValueError("Selection accepts one p-value vector")
    if criterion not in ("data", "lglk", "pcvm"):
        raise ValueError("criterion must be 'data', 'lglk', or 'pcvm'")
    threshold = scalar(threshold, "threshold")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    tolerance = _controls(algorithm, tolerance, max_iterations)
    _positive_integer(max_components, "max_components")
    _positive_integer(replicates, "replicates")
    if max_components > 10:
        raise ValueError("max_components must be at most ten")
    if not legacy_endpoints and np.any((x == 0) | (x == 1)):
        raise ValueError("Selection requires interior values unless legacy_endpoints=True")
    generator = np.random.default_rng(rng)
    previous = _fit(x, BetaMixture(1), algorithm, tolerance, max_iterations, legacy_endpoints)
    candidates: list[BetaMixtureFit] = []
    checks: list[BetaMixtureBootstrap] = []

    def result(fit: BetaMixtureFit, status: str, message: str) -> BetaMixtureSelection:
        return BetaMixtureSelection(
            fit, tuple(candidates), tuple(checks), criterion, status, message
        )

    for k in range(max_components + 1):
        try:
            current = (
                previous
                if k == 0
                else _fit(
                    x,
                    beta_mixture_start(x, previous.model),
                    algorithm,
                    tolerance,
                    max_iterations,
                    legacy_endpoints,
                )
            )
        except BetaMixtureFitError as error:
            return result(previous, "fit_failed", f"Model k={k}: {error}")
        candidates.append(current)
        if criterion == "pcvm":
            try:
                check = beta_mixture_bootstrap(
                    x,
                    current.model,
                    algorithm=algorithm,
                    replicates=replicates,
                    max_iterations=max_iterations,
                    rng=generator,
                )
            except BetaMixtureFitError as error:
                if k == 0:
                    raise
                return result(previous, "bootstrap_failed", f"Model k={k}: {error}")
            checks.append(check)
            if check.pvalue > threshold:
                return result(current, "criterion_met", f"CVM p-value exceeds {threshold}")
        elif k > 0:
            if criterion == "data":
                change = float(current.model.weights[-1])
            else:
                gain = abs(current.log_likelihood - previous.log_likelihood)
                # Uniform log likelihood is exactly zero. Avoid the source's
                # undefined 0/0 while preserving positive/0 as infinite change.
                change = (
                    gain / previous.log_likelihood
                    if previous.log_likelihood != 0
                    else (np.inf if gain > 0 else 0.0)
                )
            if change < threshold:
                return result(previous, "criterion_met", f"{criterion} change below {threshold}")
        previous = current
    return result(previous, "component_limit", f"Criterion not met through k={max_components}")
