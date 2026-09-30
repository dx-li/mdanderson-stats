"""Bounded adaptive vector-importance posterior for the six-dose Phase I/II model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import betaincc, expit, logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .parallel_phase12_model import _DOSES, _MEAN, _SD, _tally


@dataclass(frozen=True)
class Phase12ImportanceFit:
    """Posterior function summaries and integration diagnostics (no draws retained)."""

    posterior_mode: FloatArray
    proposal_covariance: FloatArray
    response_probability_mean: FloatArray
    response_probability_second_moment: FloatArray
    reference_superiority: FloatArray
    efficacy_probability: FloatArray
    future_probability: FloatArray
    pairwise_superiority: FloatArray
    toxicity_probability: FloatArray
    response_probability_mc_se: FloatArray
    response_probability_second_moment_mc_se: FloatArray
    reference_superiority_mc_se: FloatArray
    efficacy_probability_mc_se: FloatArray
    future_probability_mc_se: FloatArray
    pairwise_superiority_mc_se: FloatArray
    log_evidence: float
    integrations: int
    target_relative_error: float
    component_relative_error: FloatArray
    log_integral_mc_se: FloatArray
    ratio_mc_se: FloatArray
    converged: bool
    mode_iterations: int
    mode_gradient_norm: float


def _settings(
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    efficacy_target: float,
    future_target: float,
    toxicity_target: float,
    toxicity_prior: ArrayLike,
    integration_relative_error: float,
    max_integrations: int,
    mixture_multivariate: float,
    max_mode_iterations: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    mu, sd = finite(prior_mean, "prior_mean"), finite(prior_sd, "prior_sd")
    if mu.shape != (4,) or sd.shape != (4,) or np.any(sd <= 0):
        raise ValueError("prior_mean and positive prior_sd must have four entries")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        prior_precision = 1.0 / np.square(sd)
    if not np.all(np.isfinite(prior_precision)) or np.any(prior_precision == 0):
        raise ValueError("prior_sd gives unrepresentable prior precision")
    targets = finite([efficacy_target, future_target, toxicity_target], "targets")
    if np.any((targets <= 0) | (targets >= 1)):
        raise ValueError("targets must be in (0,1)")
    prior = finite(toxicity_prior, "toxicity_prior")
    if prior.shape != (2,) or np.any(prior <= 0) or not np.isfinite(prior.sum()):
        raise ValueError("toxicity_prior must contain two positive shapes with finite sum")
    settings = finite(
        [integration_relative_error, max_integrations, mixture_multivariate, max_mode_iterations],
        "importance settings",
    )
    if not (0 < settings[0] < 1):
        raise ValueError("integration_relative_error must lie in (0,1)")
    if (
        settings[1] != np.floor(settings[1])
        or not 100 <= settings[1] <= 1_000_000
        or int(settings[1]) % 100
    ):
        raise ValueError("max_integrations must be a multiple of 100 in 100..1000000")
    if not (0 < settings[2] < 1):
        raise ValueError("mixture_multivariate must lie strictly between zero and one")
    if settings[3] != np.floor(settings[3]) or not 1 <= settings[3] <= 10_000:
        raise ValueError("max_mode_iterations must be an integer in 1..10000")
    return mu, sd, targets, prior


def _log_prior(beta: FloatArray, mean: FloatArray, sd: FloatArray) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (beta - mean) / sd
        value = -0.5 * np.dot(z, z) - np.log(sd).sum() - 2 * np.log(2 * np.pi)
    return float(value)


def _fit_mode(
    x: FloatArray, mean: FloatArray, sd: FloatArray, max_iterations: int
) -> tuple[FloatArray, FloatArray, int, float]:
    failure, success = x[:, 0], x[:, 1]
    totals = failure + success
    precision = 1.0 / np.square(sd)

    def objective(beta: FloatArray) -> tuple[float, FloatArray]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eta = _DOSES @ beta
            p = expit(eta)
            ll = -np.sum(success * np.logaddexp(0.0, -eta) + failure * np.logaddexp(0.0, eta))
            value = -float(ll + _log_prior(beta, mean, sd))
            gradient = _DOSES.T @ (failure * p - success * expit(-eta)) + (beta - mean) * precision
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ArithmeticError("posterior mode objective is not representable")
        return value, gradient

    if np.any(totals):
        result = minimize(
            objective,
            mean.copy(),
            jac=True,
            method="BFGS",
            options={"maxiter": max_iterations, "gtol": 1e-8},
        )
        mode = np.asarray(result.x, dtype=float)
        _, gradient = objective(mode)
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if not np.all(np.isfinite(mode)) or not np.isfinite(gradient_norm):
            raise ArithmeticError("posterior mode optimizer returned nonfinite values")
        if not result.success and gradient_norm > 1e-5:
            raise ArithmeticError(
                f"posterior mode did not converge within {max_iterations} iterations"
            )
        iterations = int(result.nit)
    else:
        mode = mean.copy()
        gradient_norm = 0.0
        iterations = 0

    with np.errstate(over="ignore", invalid="ignore"):
        eta = _DOSES @ mode
        if not np.all(np.isfinite(eta)):
            raise ArithmeticError("posterior mode predictors are not representable")
        curvature = expit(eta) * expit(-eta)
        hessian = _DOSES.T @ ((totals * curvature)[:, None] * _DOSES)
    hessian.flat[::5] += precision
    try:
        covariance = np.linalg.solve(hessian, np.eye(4))
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("posterior mode covariance is not positive definite") from exc
    del factor
    if not np.all(np.isfinite(covariance)):
        raise ArithmeticError("posterior mode covariance is not representable")
    return mode, 0.5 * (covariance + covariance.T), iterations, gradient_norm


def _summary_values(beta: FloatArray, targets: FloatArray) -> FloatArray:
    eta = _DOSES @ beta
    p = expit(eta)
    values = np.empty(67)
    values[0] = 1.0  # normalizing evidence integral, as FullIntegrand component zero
    values[1:7] = 0.5
    values[2:7] = eta[1:] > eta[0]
    values[7:13] = eta >= np.log(targets[0]) - np.log1p(-targets[0])
    values[13:19] = eta > np.log(targets[1]) - np.log1p(-targets[1])
    values[19:55] = (eta[:, None] > eta[None, :]).ravel()
    values[55:61] = p * p
    values[61:67] = p
    return values


def fit_phase12_importance(
    tally: ArrayLike,
    *,
    prior_mean: ArrayLike = _MEAN,
    prior_sd: ArrayLike = _SD,
    efficacy_target: float = 0.30,
    future_target: float = 0.10,
    toxicity_target: float = 0.33,
    toxicity_prior: ArrayLike = (0.1, 0.9),
    integration_relative_error: float = 0.001,
    max_integrations: int = 10_000,
    mixture_multivariate: float = 0.99,
    max_mode_iterations: int = 1_000,
    rng: np.random.Generator,
) -> Phase12ImportanceFit:
    """Estimate six-dose posterior summaries by bounded adaptive vector importance.

    Proposals are iid from ``mixture_multivariate * N(mode, covariance)`` plus
    the remaining mass from the independent normal prior. The importance ratio
    uses likelihood times prior divided by the *full mixture density*. The
    first 60 integration components match the source indicators; six response
    means are appended as a Python summary extension. No draws are retained.
    """
    x = _tally(tally)
    mean, sd, targets, tox_prior = _settings(
        prior_mean,
        prior_sd,
        efficacy_target,
        future_target,
        toxicity_target,
        toxicity_prior,
        integration_relative_error,
        max_integrations,
        mixture_multivariate,
        max_mode_iterations,
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if max_integrations * 67 > 67_000_000:
        raise ValueError("importance integration exceeds the 67000000 component-work limit")
    mode, covariance, mode_iterations, mode_gradient_norm = _fit_mode(
        x, mean, sd, int(max_mode_iterations)
    )
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("proposal covariance is not positive definite") from exc
    log_det_cov = 2.0 * float(np.log(np.diag(chol)).sum())
    failure, success = x[:, 0], x[:, 1]
    log_mix = np.log([mixture_multivariate, 1.0 - mixture_multivariate])

    def log_likelihood(beta: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            eta = _DOSES @ beta
            result = -float(
                np.sum(success * np.logaddexp(0.0, -eta) + failure * np.logaddexp(0.0, eta))
            )
        if not np.isfinite(result):
            raise ArithmeticError("response log likelihood is not representable")
        return result

    def log_proposal(beta: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            centered = beta - mode
            whitened = np.linalg.solve(chol, centered)
            mvn = -2 * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * np.dot(whitened, whitened)
        prior = _log_prior(beta, mean, sd)
        return float(logsumexp(log_mix + [mvn, prior]))

    n = 0
    log_scale = -np.inf
    average = np.zeros(67)
    m2 = np.zeros(67)
    evidence_cross = np.zeros(67)
    raw_relative_error = np.full(67, np.inf)
    converged = False
    while n < int(max_integrations):
        if rng.random() < mixture_multivariate:
            beta = mode + chol @ rng.normal(size=4)
        else:
            beta = rng.normal(mean, sd)
        log_weight = log_likelihood(beta) + _log_prior(beta, mean, sd) - log_proposal(beta)
        if not np.isfinite(log_weight):
            raise ArithmeticError("importance log weight is not representable")
        if log_weight > log_scale:
            if np.isfinite(log_scale):
                factor = np.exp(log_scale - log_weight)
                average *= factor
                m2 *= factor * factor
                evidence_cross *= factor * factor
            log_scale = log_weight
        weight = float(np.exp(log_weight - log_scale))
        observation = weight * _summary_values(beta, targets)
        n += 1
        delta = observation - average
        average += delta / n
        delta_after = observation - average
        m2 += delta * delta_after
        evidence_cross += delta * delta_after[0]
        if n % 100 == 0:
            sample_se = np.sqrt(np.maximum(m2, 0.0) / (n * (n - 1)))
            scale = np.maximum(average, 0.01 * average[0])
            raw_relative_error = np.divide(
                sample_se,
                average,
                out=np.full_like(sample_se, np.inf),
                where=average > 0,
            )
            # Index zero is evidence; the 60 original summaries occupy 1:61.
            # Six appended response means are a Python extension and do not alter
            # the native stopping test.
            converged = bool(np.all(sample_se[:61] <= integration_relative_error * scale[:61]))
            if converged:
                break

    if average[0] <= 0 or not np.isfinite(average[0]):
        raise ArithmeticError("importance integration did not estimate positive evidence")
    ratios = average[1:] / average[0]
    variance = np.maximum(m2[1:] + np.square(ratios) * m2[0] - 2 * ratios * evidence_cross[1:], 0.0)
    ratio_se = np.sqrt(variance / (n * (n - 1))) / average[0]
    integral_se = np.sqrt(np.maximum(m2, 0.0) / (n * (n - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_integral_se = np.where(integral_se > 0, log_scale + np.log(integral_se), -np.inf)
    log_evidence = log_scale + float(np.log(average[0]))
    if not np.isfinite(log_evidence) or not np.all(np.isfinite(ratios)):
        raise ArithmeticError("importance posterior summaries are not representable")

    pairwise = ratios[18:54].reshape(6, 6)
    return Phase12ImportanceFit(
        _freeze(mode),
        _freeze(covariance),
        _freeze(ratios[60:66]),
        _freeze(ratios[54:60]),
        _freeze(ratios[0:6]),
        _freeze(ratios[6:12]),
        _freeze(ratios[12:18]),
        _freeze(pairwise),
        _freeze(betaincc(tox_prior[0] + x[:, 3], tox_prior[1] + x[:, 2], targets[2])),
        _freeze(ratio_se[60:66]),
        _freeze(ratio_se[54:60]),
        _freeze(ratio_se[0:6]),
        _freeze(ratio_se[6:12]),
        _freeze(ratio_se[12:18]),
        _freeze(ratio_se[18:54].reshape(6, 6)),
        log_evidence,
        n,
        float(integration_relative_error),
        _freeze(raw_relative_error),
        _freeze(log_integral_se),
        _freeze(ratio_se),
        converged,
        mode_iterations,
        mode_gradient_norm,
    )
