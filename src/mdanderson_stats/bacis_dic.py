"""Deterministic DIC for the BaCIS first-stage classification model.

The inspected bacistool diagnostic scores the independent two-component
logistic-normal model, not the later within-cluster borrowing fit. Its Plummer
pD population target reduces to ``n * Cov(p, logit(p))`` under the posterior.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit, gammaln

from ._validation import FloatArray
from .bacis import _readonly, _validate_data, bacis_classify


@dataclass(frozen=True)
class BaCISClassificationDIC:
    """Per-subgroup DIC components and the total first-stage classification DIC."""

    posterior_mean: FloatArray
    posterior_logit_mean: FloatArray
    mean_deviance: FloatArray
    penalty: FloatArray
    dic: FloatArray
    total_dic: float
    high_component_probability: FloatArray
    quadrature_error: FloatArray


def _component_moments(
    successes: int,
    trials: int,
    location: float,
    precision: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Posterior moments from a curvature-scaled integral in standard coordinates."""

    def score(eta: float) -> float:
        return successes - trials * expit(eta) - precision * (eta - location)

    left = location + (successes - trials) / precision
    right = location + successes / precision
    mode = brentq(score, left, right, xtol=1e-12, rtol=1e-14)
    p_mode = float(expit(mode))
    scale = 1.0 / np.sqrt(precision + trials * p_mode * (1.0 - p_mode))
    peak = (
        -successes * np.logaddexp(0.0, -mode)
        - (trials - successes) * np.logaddexp(0.0, mode)
        - 0.5 * precision * (mode - location) ** 2
    )

    failure_mode = float(expit(-mode))

    def integrand(z: float) -> FloatArray:
        eta = mode + scale * z
        log_kernel = (
            -successes * np.logaddexp(0.0, -eta)
            - (trials - successes) * np.logaddexp(0.0, eta)
            - 0.5 * precision * (eta - location) ** 2
        )
        weight = float(np.exp(log_kernel - peak))
        if mode > 0.0:
            # q_mode - q_eta equals p_eta - p_mode without subtracting numbers
            # rounded near one.
            delta_probability = failure_mode - float(expit(-eta))
        else:
            delta_probability = float(expit(eta)) - p_mode
        return np.asarray(
            [
                weight,
                z * weight,
                delta_probability * weight,
                z * delta_probability * weight,
                -np.logaddexp(0.0, -eta) * weight,
                -np.logaddexp(0.0, eta) * weight,
            ],
            dtype=float,
        )

    integral, error, info = quad_vec(
        integrand,
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=2e-11,
        norm="max",
        cache_size=1_048_576,
        limit=300,
        workers=1,
        full_output=True,
    )
    values = np.asarray(integral, dtype=float)
    norm = float(values[0])
    if not info.success or not np.isfinite(values).all() or not np.isfinite(error) or norm <= 0:
        raise ArithmeticError(f"BaCIS DIC posterior quadrature failed: {info.message}")
    relative_error = float(error / norm)
    if relative_error > 2e-7:
        raise ArithmeticError("BaCIS DIC quadrature did not meet its error tolerance")
    z_mean = float(values[1] / norm)
    mean_delta_probability = float(values[2] / norm)
    if mode > 0.0:
        failure_mean = failure_mode - mean_delta_probability
        probability_mean = 1.0 - failure_mean
    else:
        probability_mean = p_mode + mean_delta_probability
        failure_mean = 1.0 - probability_mean
    eta_mean = float(mode + scale * z_mean)
    covariance = float(scale * (values[3] / norm - z_mean * mean_delta_probability))
    mean_log_p = float(values[4] / norm)
    mean_log_failure = float(values[5] / norm)
    if not np.isfinite(
        [eta_mean, probability_mean, failure_mean, covariance, mean_log_p, mean_log_failure]
    ).all():
        raise ArithmeticError("BaCIS DIC posterior moments are not finite")
    return (
        eta_mean,
        probability_mean,
        failure_mean,
        covariance,
        mean_log_p,
        mean_log_failure,
        relative_error,
    )


def bacis_classification_dic(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    phi_low: float = 0.1,
    phi_high: float = 0.3,
    classification_precision: float | None = None,
) -> BaCISClassificationDIC:
    """Compute deterministic first-stage BaCIS DIC with Plummer's pD penalty.

    Each subgroup's posterior is the equal-prior mixture of the low/high
    logistic-normal components used by :func:`bacis_classify`. The adaptive
    classification cutoff does not alter this posterior and is not an input.
    ``mean_deviance`` includes the full binomial coefficient. The penalty is
    ``n * Cov(p, logit(p))``, including between-component mixture covariance.
    Quadrature errors are relative component-integral estimates, with columns
    ordered low then high. Failed integrations raise rather than returning an
    incomplete criterion.
    """
    classification = bacis_classify(
        successes,
        trials,
        phi_low=phi_low,
        phi_high=phi_high,
        classification_precision=classification_precision,
    )
    # Validation, prior centers, and precision match the classifier.
    y, n = _validate_data(successes, trials)
    locations = np.log(np.array([phi_low, phi_high])) - np.log1p(-np.array([phi_low, phi_high]))
    weights = np.column_stack((classification.low_probability, classification.high_probability))
    groups = y.size
    mean_p = np.empty(groups)
    mean_eta = np.empty(groups)
    mean_deviance = np.empty(groups)
    penalty = np.empty(groups)
    errors = np.empty((groups, 2))
    log_choose = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)

    for i in range(groups):
        component = [
            _component_moments(int(y[i]), int(n[i]), float(locations[k]), classification.precision)
            for k in range(2)
        ]
        w = weights[i]
        component_p = np.array([item[1] for item in component])
        component_q = np.array([item[2] for item in component])
        component_eta = np.array([item[0] for item in component])
        component_mean = float(w @ component_p)
        component_failure = float(w @ component_q)
        mean_p[i] = 1.0 - component_failure if component_mean > 0.5 else component_mean
        mean_eta[i] = float(w @ component_eta)
        within = float(w @ np.array([item[3] for item in component]))
        if np.all(component_p > 0.5):
            probability_difference = component_q[0] - component_q[1]
        elif np.all(component_p <= 0.5):
            probability_difference = component_p[1] - component_p[0]
        else:
            probability_difference = component_p[1] - component_p[0]
        between = float(
            w[0] * w[1] * probability_difference * (component_eta[1] - component_eta[0])
        )
        covariance = within + between
        scale_cov = max(abs(within), abs(between), np.finfo(float).tiny)
        if covariance < -128 * np.finfo(float).eps * scale_cov:
            raise ArithmeticError("BaCIS DIC posterior covariance is materially negative")
        penalty[i] = int(n[i]) * max(0.0, covariance)
        mean_log_p = float(w @ np.array([item[4] for item in component]))
        mean_log_failure = float(w @ np.array([item[5] for item in component]))
        mean_deviance[i] = -2.0 * (
            log_choose[i] + y[i] * mean_log_p + (n[i] - y[i]) * mean_log_failure
        )
        errors[i] = np.maximum(classification.quadrature_error[i], [item[6] for item in component])

    dic = mean_deviance + penalty
    if not np.isfinite(
        np.concatenate((mean_p, mean_eta, mean_deviance, penalty, dic, errors.ravel()))
    ).all():
        raise ArithmeticError("BaCIS DIC result is not finite")
    return BaCISClassificationDIC(
        _readonly(mean_p),
        _readonly(mean_eta),
        _readonly(mean_deviance),
        _readonly(penalty),
        _readonly(dic),
        float(np.sum(dic)),
        _readonly(weights[:, 1]),
        _readonly(errors),
    )
