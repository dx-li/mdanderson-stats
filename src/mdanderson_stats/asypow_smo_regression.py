"""SMO expected-likelihood power for binary, Poisson and survival regression."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
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
    """Linear/quadratic logistic/cloglog, Poisson or exponential-survival SMO.

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
    if family not in ("logistic", "cloglog", "poisson", "exponential"):
        raise ValueError("family must be logistic, cloglog, poisson, or exponential")
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
    log_weight = np.log(count[used]) + np.log(relative[group_index])
    log_weight -= logsumexp(log_weight)
    return _fit_regression(
        theta,
        design,
        group_index,
        log_weight,
        family,
        length,
        constraints,
        lower,
        upper,
        subtract_df,
        tolerance,
    )


def _fit_regression(
    theta: FloatArray,
    design: FloatArray,
    group_index: NDArray[np.intp],
    log_weight: FloatArray,
    family: str,
    length: FloatArray | None,
    constraints: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    subtract_df: bool,
    tolerance: float,
) -> SMOPower:
    groups, columns = theta.shape
    for g in range(groups):
        block = design[group_index == g]
        scale = np.max(np.abs(block), axis=0)
        if np.any(scale == 0) or np.linalg.matrix_rank(block / scale) < columns:
            raise ValueError("each group's observed design must identify all coefficients")

    def predictors(q: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            eta = np.sum(design * q.reshape(theta.shape)[group_index], axis=1)
        return finite(eta, "linear predictors")

    def binary_logs(eta: FloatArray) -> tuple[FloatArray, FloatArray]:
        if family == "loglinear":
            if np.any(eta >= 0):
                raise ValueError(
                    "log-linear binomial predictors must be negative; review bounds/design"
                )
            return eta, np.log(-np.expm1(eta))
        if family != "cloglog":
            return -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            hazard = np.exp(eta)
            first = np.where(eta < -36, eta, np.log(-np.expm1(-hazard)))
        return finite(first, "cloglog log probabilities"), finite(-hazard, "cloglog log survival")

    eta = predictors(theta.ravel())
    log_p, log_s = binary_logs(eta)
    # Normalize likelihood curvature for optimization; restore its original
    # scale in the returned per-observation divergence (essential for rare means).
    log_information = (
        log_p - log_s
        if family == "loglinear"
        else _log_information(eta, family, None if length is None else length[group_index])
    )
    log_scale = float(logsumexp(log_weight + log_information))
    if not np.isfinite(log_scale):
        raise ArithmeticError("alternative predictor information is numerically unresolved")

    def likelihood(_p: FloatArray, q: FloatArray) -> float:
        candidate = predictors(q)
        if family in ("poisson", "exponential"):
            log_kl = (log_weight + log_information - log_scale) + _log_exp_remainder(
                candidate - eta
            )
        else:
            candidate_p, candidate_s = binary_logs(candidate)
            first = (log_weight + log_p - log_scale) + _log_exp_remainder(candidate_p - log_p)
            second = (log_weight + log_s - log_scale) + _log_exp_remainder(candidate_s - log_s)
            log_kl = np.logaddexp(first, second)
        with np.errstate(over="ignore", under="ignore"):
            return -float(np.exp(logsumexp(log_kl)))

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
        elif family == "loglinear":
            _, candidate_s = binary_logs(candidate)
            log_score = log_weight + log_p - candidate_s + log_difference - log_scale
        elif family == "cloglog":
            # p_alt-p_null = exp(-z_null)-exp(-z_alt), z=exp(eta).
            # Preserve the difference even when z or the probabilities underflow.
            log_gap = eta + log_difference
            log_drop = np.zeros_like(log_gap)
            small = log_gap < -36
            log_drop[small] = log_gap[small]
            ordinary = ~small & (log_gap < 36)
            log_drop[ordinary] = np.log(-np.expm1(-np.exp(log_gap[ordinary])))
            candidate_p, _ = binary_logs(candidate)
            with np.errstate(under="ignore"):
                minimum_hazard = np.exp(np.minimum(eta, candidate))
            log_score = log_weight + candidate - candidate_p - minimum_hazard + log_drop - log_scale
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
