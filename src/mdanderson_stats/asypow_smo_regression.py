"""SMO expected-likelihood power for binary, Poisson and survival regression."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .asypow_generic import asypow_smo_generic
from .asypow_regression import _log_information
from .asypow_smo import SMOPower


def _log_exp_remainder(x: FloatArray) -> FloatArray:
    """Log(expm1(x)-x), retaining quadratic terms near zero."""
    result = np.empty_like(x)
    small = np.abs(x) < 0.125
    z = x[small]
    polynomial = np.full_like(z, 1 / 479001600)
    for factorial in [39916800, 3628800, 362880, 40320, 5040, 720, 120, 24, 6, 2]:
        polynomial = 1 / factorial + z * polynomial
    with np.errstate(divide="ignore", under="ignore"):
        result[small] = 2 * np.log(np.abs(z)) + np.log(polynomial)
        large = x > 50
        result[large] = x[large] + np.log1p(-(1 + x[large]) * np.exp(-x[large]))
        ordinary = ~small & ~large
        result[ordinary] = np.log(np.expm1(x[ordinary]) - x[ordinary])
    return result


def asypow_smo_regression(
    parameters: ArrayLike,
    covariates: ArrayLike,
    *,
    constraints: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    family: str = "logistic",
    duration: ArrayLike | None = None,
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
    tolerance: float = 1e-8,
) -> SMOPower:
    """Linear/quadratic logistic, log-mean Poisson or log-rate survival SMO.

    Rows contain intercept, slope and optional quadratic coefficient. Original
    one-based constraints index flattened coefficient rows. Positive design
    observations are weighted by group_size before normalization over all rows.
    Bounds are computational coefficient bounds, as in asypow_smo_generic.
    """
    theta = finite(parameters, "parameters")
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[1] not in (2, 3) or not 1 <= theta.size <= 500:
        raise ValueError("parameters need 2 or 3 columns and at most 500 coefficients")
    if family not in ("logistic", "poisson", "exponential"):
        raise ValueError("family must be logistic, poisson, or exponential")
    groups, columns = theta.shape
    length = None
    if family == "exponential":
        if duration is None:
            raise ValueError("duration is required for exponential survival")
        length = np.broadcast_to(finite(duration, "duration"), (groups,))
        if np.any(length <= 0):
            raise ValueError("duration must be positive")
    elif duration is not None:
        raise ValueError("duration is only used for exponential survival")
    x = finite(covariates, "covariates")
    if x.ndim == 1:
        x = np.broadcast_to(x, (groups, x.size))
    if x.ndim != 2 or x.shape[0] != groups or not 1 <= x.size <= 1_000_000:
        raise ValueError("covariates need one row per group and 1..1000000 entries")
    count = np.broadcast_to(finite(observations, "observations"), x.shape)
    relative = np.broadcast_to(finite(group_size, "group_size"), (groups,))
    if np.any(count < 0) or np.any(relative <= 0) or np.any(np.sum(count > 0, axis=1) == 0):
        raise ValueError("group sizes must be positive; each group needs positive observations")
    used = count > 0
    group_index = np.nonzero(used)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        design = np.vander(x[used], N=columns, increasing=True)
    if not np.all(np.isfinite(design)):
        raise ArithmeticError("polynomial design overflow; rescale covariates")
    for g in range(groups):
        block = design[group_index == g]
        scale = np.max(np.abs(block), axis=0)
        if np.any(scale == 0) or np.linalg.matrix_rank(block / scale) < columns:
            raise ValueError("each group's observed design must identify all coefficients")
    log_weight = np.log(count[used]) + np.log(relative[group_index])
    log_weight -= logsumexp(log_weight)

    def predictors(q: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            eta = np.sum(design * q.reshape(theta.shape)[group_index], axis=1)
        return finite(eta, "linear predictors")

    eta = predictors(theta.ravel())
    log_p = -np.logaddexp(0, -eta)
    log_s = -np.logaddexp(0, eta)
    # Normalize likelihood curvature for optimization; restore its original
    # scale in the returned per-observation divergence (essential for rare means).
    log_information = _log_information(eta, family, None if length is None else length[group_index])
    log_scale = float(logsumexp(log_weight + log_information))

    def likelihood(_p: FloatArray, q: FloatArray) -> float:
        candidate = predictors(q)
        if family in ("poisson", "exponential"):
            log_kl = log_information + _log_exp_remainder(candidate - eta)
        else:
            first = log_p + _log_exp_remainder(-np.logaddexp(0, -candidate) - log_p)
            second = log_s + _log_exp_remainder(-np.logaddexp(0, candidate) - log_s)
            log_kl = np.logaddexp(first, second)
        with np.errstate(over="ignore", under="ignore"):
            return -float(np.exp(logsumexp(log_weight + log_kl) - log_scale))

    def gradient(_p: FloatArray, q: FloatArray) -> FloatArray:
        candidate = predictors(q)
        delta = candidate - eta
        with np.errstate(divide="ignore", under="ignore"):
            log_difference = np.empty_like(delta)
            large = delta > 50
            log_difference[large] = delta[large] + np.log1p(-np.exp(-delta[large]))
            log_difference[~large] = np.log(np.abs(np.expm1(delta[~large])))
        if family in ("poisson", "exponential"):
            log_score = log_weight + log_information + log_difference - log_scale
        else:
            log_score = log_weight + log_p - np.logaddexp(0, candidate) + log_difference - log_scale
        with np.errstate(over="ignore", under="ignore"):
            score = -np.sign(delta) * np.exp(log_score)
        result = np.zeros_like(theta)
        np.add.at(result, group_index, score[:, None] * design)
        return finite(result.ravel(), "regression gradient")

    fitted = asypow_smo_generic(
        theta,
        likelihood,
        gradient=gradient,
        lower=lower,
        upper=upper,
        constraints=constraints,
        subtract_df=subtract_df,
        tolerance=tolerance,
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        w = float(np.exp(np.log(fitted.divergence_per_observation) + log_scale))
    if not np.isfinite(w):
        raise ArithmeticError("regression SMO divergence is not representable")
    return SMOPower(w, fitted.degrees_of_freedom, fitted.null_parameters, fitted.subtract_df)
