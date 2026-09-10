"""Noise-free expectation of the native BayesESS survival information criterion."""

from dataclasses import dataclass

import numpy as np

from ._validation import FloatArray, scalar
from .bayesian_monitoring import _integer
from .boin import _owned
from .regression_ess import _ordinary


@dataclass(frozen=True)
class SurvivalPriorESS:
    log_ess: float
    grid_ess: float
    at_search_boundary: bool
    patients: FloatArray
    normalized_information_distance: FloatArray

    @property
    def ess(self) -> float:
        """Unconstrained continuous root; raises outside positive float64 range."""
        return float(_ordinary(self.log_ess))


def survival_prior_ess(
    shape: float, scale: float, *, censoring_time: float = 3, max_patients: int = 100
) -> SurvivalPriorESS:
    """Analytic expectation of BayesESS::essSurv for an exponential mean IG prior.

    This reproduces the native criterion's expectation, not its random sequence.
    Shape must exceed one so its prior mean exists. The source fixes censoring
    time to 3; this interface allows a different common administrative window.
    Grid ESS uses the native 50-point linear-interpolation grid on [1,max_patients].
    Prior and posterior curvature use different evaluation points in the source;
    this is not a general implementation of the fixed-point Morita ESS definition.
    """
    a, b, c = (
        scalar(shape, "shape"),
        scalar(scale, "scale"),
        scalar(censoring_time, "censoring_time"),
    )
    maximum = _integer(max_patients, "max_patients")
    if not 1 < a <= 1e6 or b <= 0 or c <= 0 or not 2 <= maximum <= 100000:
        raise ValueError("require shape in (1,1e6], positive scale/window, max_patients 2..100000")
    # Lambda=1/mu ~ Gamma(a, rate=b). E[lambda^2]=a(a+1)/b^2.
    # Per-observation expected curvature / E[lambda^2] is
    # f = 1 - (b/(b+c))^(a+2). Work in logs even for a tiny window/scale ratio.
    ratio = np.log(c) - np.log(b)
    log_softplus = ratio if ratio < -36 else np.log(np.logaddexp(0, ratio))
    z = np.log(a + 2) + log_softplus
    log_fraction = z if z < -36 else 0.0 if z > 7 else np.log(-np.expm1(-np.exp(z)))
    prior_ratio = (a - 3) * ((a - 1) / a) * ((a - 1) / (a + 1))
    log_ess = float(np.log1p(prior_ratio) - log_fraction)
    grid = np.linspace(1, maximum, 50)
    root = float(np.exp(np.clip(log_ess, 0, np.log(maximum))))
    selected = int(np.argmin(np.abs(grid - root)))
    distance = prior_ratio + 1 - grid * np.exp(log_fraction)
    return SurvivalPriorESS(
        log_ess, float(grid[selected]), selected in (0, 49), _owned(grid), _owned(distance)
    )
