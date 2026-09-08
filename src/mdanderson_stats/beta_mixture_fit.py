"""MULTI's S STBETA initialization and EMBETA exponential-family EM fit."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import digamma, logsumexp, polygamma

from ._validation import FloatArray, scalar
from .beta_mixture import BetaMixture, _logs
from .multiplicity import _pvalues


class BetaMixtureFitError(ValueError):
    """The requested mixture fit could not be completed or is degenerate."""


@dataclass(frozen=True)
class BetaMixtureFit:
    model: BetaMixture
    log_likelihood: float
    cramer_von_mises: float
    iterations: int
    log_likelihood_history: FloatArray


def beta_mixture_start(pvalues: ArrayLike, previous: BetaMixture | None = None) -> BetaMixture:
    """STBETA: add a beta component to observations below the previous CDF's .05.

    The first component uses p<.05. At least three selected observations and
    positive method-of-moments shapes are required. Uses sample variance (ddof=1).
    """
    x = _pvalues(pvalues)
    if x.ndim != 1:
        raise ValueError("Mixture initialization accepts one p-value vector")
    previous = BetaMixture(1) if previous is None else previous
    selected = x[previous.cdf(x) < 0.05]
    if selected.size < 3:
        raise BetaMixtureFitError("STBETA requires three observations below the previous CDF's .05")
    mean, variance = float(selected.mean()), float(selected.var(ddof=1))
    if not 0 < variance < mean * (1 - mean):
        raise BetaMixtureFitError("Selected observations do not yield positive beta moment shapes")
    concentration = mean * (1 - mean) / variance - 1
    weight = selected.size / x.size
    return BetaMixture(
        previous.null_weight * (1 - weight),
        np.r_[previous.weights * (1 - weight), weight],
        np.r_[previous.a, mean * concentration],
        np.r_[previous.b, (1 - mean) * concentration],
    )


def _shapes_from_log_moments(target: FloatArray, initial: FloatArray) -> FloatArray:
    """SWPPAR's positive Newton updates, with a floating-point residual check."""
    shapes = initial.copy()
    for _ in range(10000):
        combined = shapes.sum()
        psi_shapes, psi_total = digamma(shapes), float(digamma(combined))
        residual = psi_shapes - psi_total - target
        # Subtracting nearly equal digammas limits attainable residual precision.
        # The original 1e-20 threshold can cycle forever for concentrated betas.
        rounding = 8 * np.finfo(float).eps * np.maximum(1, np.abs(psi_shapes) + abs(psi_total))
        if np.all(np.abs(residual) <= rounding):
            return shapes
        cross = -float(polygamma(1, combined))
        jacobian = np.diag(polygamma(1, shapes)) + cross
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as error:
            raise BetaMixtureFitError("Singular beta log-moment equations") from error
        if not np.all(np.isfinite(step)):
            raise BetaMixtureFitError("Nonfinite beta shape update")
        for _ in range(1075):
            if np.all(shapes + step > 1e-10):
                break
            step *= 0.5
        else:
            raise BetaMixtureFitError("No positive beta shape update is possible")
        shapes += step
        if np.sum(np.abs(step)) <= 1e-6:
            return shapes
    raise BetaMixtureFitError("Beta log-moment solver exceeded 10000 iterations")


def fit_beta_mixture_em(
    pvalues: ArrayLike,
    initial: BetaMixture | None = None,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 100000,
    legacy_endpoints: bool = False,
) -> BetaMixtureFit:
    """Fit MULTI's uniform-plus-beta model by EMBETA's EM updates.

    Defaults to STBETA's one-component start. Supply beta_mixture_start(data,
    previous_fit.model) to add a component, or an explicit initial model.
    Requires n>3*k+1, as in betamix.k. Endpoints are rejected unless the archived
    INITLN approximation is explicitly requested. Convergence measures changes
    in null weight and component log moments, as the source does; it does not
    imply that a global likelihood maximum has been found.
    """
    x = _pvalues(pvalues)
    if x.ndim != 1:
        raise ValueError("Mixture fitting accepts one p-value vector")
    tolerance = scalar(tolerance, "tolerance")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between zero and one")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    model = beta_mixture_start(x) if initial is None else initial
    k = model.weights.size
    if x.size <= 3 * k + 1:
        raise BetaMixtureFitError("Mixture fitting requires n>3*k+1")
    if k == 0:
        return BetaMixtureFit(model, 0.0, float(model.cramer_von_mises(x)), 0, np.array([0.0]))
    if not legacy_endpoints and np.any((x == 0) | (x == 1)):
        raise ValueError("EM requires interior p-values unless legacy_endpoints=True")
    lx, ly = _logs(x, legacy_endpoints)
    log_data = np.column_stack((lx, ly))
    shapes = np.column_stack((model.a, model.b))
    phi = digamma(shapes) - digamma(shapes.sum(axis=1))[:, None]
    terms = model._log_terms(x, legacy_endpoints)
    total = logsumexp(terms, axis=-1)
    history = [float(total.sum())]
    if not np.all(np.isfinite(total)):
        raise BetaMixtureFitError("Initial model has nonfinite log likelihood")
    for iteration in range(1, max_iterations + 1):
        responsibility = np.exp(terms - total[:, None])
        masses = responsibility.sum(axis=0)
        if np.any(masses[1:] <= 100 * np.finfo(np.float64).tiny):
            raise BetaMixtureFitError("Not enough effective weight in a beta component")
        weights = masses / x.size
        next_phi = (responsibility[:, 1:].T @ log_data) / masses[1:, None]
        distance = np.sqrt((model.null_weight - weights[0]) ** 2 + np.sum((next_phi - phi) ** 2))
        for j in range(k):
            shapes[j] = _shapes_from_log_moments(next_phi[j], shapes[j])
        model = BetaMixture(float(weights[0]), weights[1:], shapes[:, 0], shapes[:, 1])
        phi = next_phi
        terms = model._log_terms(x, legacy_endpoints)
        total = logsumexp(terms, axis=-1)
        likelihood = float(total.sum())
        if not np.isfinite(likelihood) or likelihood < history[-1] - 1e-8 * max(
            1, abs(history[-1])
        ):
            raise BetaMixtureFitError("EM update failed its likelihood check")
        history.append(likelihood)
        if distance <= tolerance:
            return BetaMixtureFit(
                model, likelihood, float(model.cramer_von_mises(x)), iteration, np.array(history)
            )
    raise BetaMixtureFitError(f"EM exceeded {max_iterations} iterations without convergence")
